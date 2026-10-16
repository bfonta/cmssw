#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/HcalObjects/interface/HcalPFCuts.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/BarrelSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/BarrelSoARecHitsDeviceCollection.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoParticleFlow/PFClusterProducer/interface/CaloRecHitResolutionProvider.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Converts the PF rechits of the ECAL (EB) or HCAL (HB) barrel into the barrel rechit SoA, grouped by layer, like
  // BarrelCLUEAlgoT::populate: (dim1, dim2, dim3) = (eta, phi in [0, 2pi), distance from the origin), the noise from
  // the PF rechit thresholds, and the time error from the energy, as in BarrelLayerClusterProducer. Also produces the
  // number of rechits of each non-empty layer, the batch sizes of the downstream clustering.
  class BarrelSoARecHitsProducer : public stream::EDProducer<> {
  public:
    BarrelSoARecHitsProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          isEcal_(config.getParameter<std::string>("detector") == "EB"),
          maxLayerIndex_(config.getParameter<unsigned int>("maxLayerIndex")),
          timeResolutionCalc_(config.getParameterSet("timeResolutionCalc")),
          ticlGeomToken_(
              consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>(edm::ESInputTag("", "withBarrel"))),
          ticlGeomLookupToken_(consumesCollector().esConsumes<TICLGeomLookupHost, CaloGeometryRecord>(
              edm::ESInputTag("", "withBarrel"))),
          ticlGeomLayersToken_(
              consumesCollector().esConsumes<TICLGeomLayersHost, CaloGeometryRecord>(edm::ESInputTag("", ""))),
          ecalThresholdsToken_(consumesCollector().esConsumes<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>()),
          hcalCutsToken_(consumesCollector().esConsumes<HcalPFCuts, HcalPFCutsRcd>()),
          hitsToken_(consumes<reco::PFRecHitCollection>(config.getParameter<edm::InputTag>("recHits"))),
          deviceToken_{produces()},
          layerSizesToken_{produces("layerSizes")} {}

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      rhtools_.setGeometry(
          iSetup.getData(ticlGeomToken_), iSetup.getData(ticlGeomLookupToken_), iSetup.getData(ticlGeomLayersToken_));
      auto const& ecalThresholds = iSetup.getData(ecalThresholdsToken_);
      auto const& hcalCuts = iSetup.getData(hcalCutsToken_);
      auto const& hits = iEvent.get(hitsToken_);

      // The rechit SoA is emitted layer-contiguous. Unlike HGCAL, all the PF rechits are kept: they already passed
      // the PF thresholds.
      const unsigned int numberOfLayers = maxLayerIndex_ + 1;
      std::vector<uint32_t> hitsPerLayer(numberOfLayers, 0);
      for (const reco::PFRecHit& hit : hits) {
        hitsPerLayer[layer(hit.detId())]++;
      }

      std::vector<uint32_t> layerCursor(numberOfLayers, 0);
      std::vector<uint32_t> layerSizes;
      layerSizes.reserve(numberOfLayers);
      uint32_t nextLayerStart = 0;
      for (unsigned int l = 0; l < numberOfLayers; ++l) {
        layerCursor[l] = nextLayerStart;
        nextLayerStart += hitsPerLayer[l];
        if (hitsPerLayer[l] > 0)
          layerSizes.push_back(hitsPerLayer[l]);
      }

      BarrelSoARecHitsHostCollection cells(iEvent.queue(), hits.size());
      auto cellsView = cells.view();
      // Hits are written grouped by layer, in increasing layer order, and keep their relative order within a layer
      for (unsigned int i = 0; i < hits.size(); ++i) {
        const reco::PFRecHit& hit = hits[i];
        const DetId detid(hit.detId());
        const unsigned int hitLayer = layer(detid);
        const GlobalPoint position(rhtools_.getPosition(detid));
        auto entryInSoA = cellsView[layerCursor[hitLayer]++];

        entryInSoA.dim1() = position.eta();
        // CLUEstering wraps phi over [0, 2pi)
        float phi = position.phi();
        if (phi < 0.f) {
          phi += 2.f * static_cast<float>(M_PI);
        }
        entryInSoA.dim2() = phi;
        // The distance from the origin, as in BarrelCLUEAlgoT: with eta and phi, it gives the cartesian position
        entryInSoA.dim3() = position.mag();
        entryInSoA.layer() = hitLayer;
        entryInSoA.energy() = hit.energy();
        entryInSoA.mipEnergy() = hit.energy();
        entryInSoA.sigmaNoise() = isEcal_ ? ecalThresholds[detid] : hcalCuts.getValues(detid)->seedThreshold();
        entryInSoA.recHitIndex() = i;
        entryInSoA.flags() = HGCalRecHitFlags::kNormal;
        entryInSoA.detid() = detid.rawId();
        entryInSoA.time() = hit.time();
        // BarrelLayerClusterProducer weights the rechit times by 1 / timeResolution2(energy), the layer cluster
        // kernels by 1 / timeError^2
        entryInSoA.timeError() = std::sqrt(timeResolutionCalc_.timeResolution2(hit.energy()));
      }

      if constexpr (!std::is_same_v<Device, alpaka_common::DevHost>) {
        BarrelSoARecHitsDeviceCollection deviceProduct{iEvent.queue(), cells->metadata().size()};
        alpaka::memcpy(iEvent.queue(), deviceProduct.buffer(), cells.const_buffer());
        iEvent.emplace(deviceToken_, std::move(deviceProduct));
      } else {
        iEvent.emplace(deviceToken_, std::move(cells));
      }

      iEvent.emplace(layerSizesToken_, std::move(layerSizes));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.ifValue(edm::ParameterDescription<std::string>(
                       "detector", "EB", true, edm::Comment("the barrel calorimeter: EB (ECAL) or HB (HCAL)")),
                   edm::allowedValues<std::string>("EB", "HB"));
      desc.add<edm::InputTag>("recHits", edm::InputTag("particleFlowRecHitECAL"));
      desc.add<unsigned int>("maxLayerIndex", 0)
          ->setComment("Highest layer: 0 for EB (one layer), the highest depth for HB (depth + 1 for HO).");

      edm::ParameterSetDescription timeResolutionCalcDesc;
      timeResolutionCalcDesc.add<double>("noiseTerm", 1.10889);
      timeResolutionCalcDesc.add<double>("constantTerm", 0.428192);
      timeResolutionCalcDesc.add<double>("corrTermLowE", 0.0510871);
      timeResolutionCalcDesc.add<double>("threshLowE", 0.5);
      timeResolutionCalcDesc.add<double>("constantTermLowE", 0.0);
      timeResolutionCalcDesc.add<double>("noiseTermLowE", 1.31883);
      timeResolutionCalcDesc.add<double>("threshHighE", 5.0);
      desc.add<edm::ParameterSetDescription>("timeResolutionCalc", timeResolutionCalcDesc);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    bool const isEcal_;
    unsigned int const maxLayerIndex_;
    CaloRecHitResolutionProvider timeResolutionCalc_;
    ticlgeom::Tools rhtools_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> const ticlGeomToken_;
    edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> const ticlGeomLookupToken_;
    edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> const ticlGeomLayersToken_;
    edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> const ecalThresholdsToken_;
    edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> const hcalCutsToken_;
    edm::EDGetTokenT<reco::PFRecHitCollection> const hitsToken_;
    device::EDPutToken<BarrelSoARecHitsDeviceCollection> const deviceToken_;
    edm::EDPutTokenT<std::vector<uint32_t>> const layerSizesToken_;

    // The layer of a rechit, as in BarrelCLUEAlgoT::populate: 0 for ECAL, the depth for HCAL (plus one for HO)
    unsigned int layer(const DetId& detid) const {
      unsigned int layer = 0;
      if (detid.det() == DetId::Hcal) {
        const HcalDetId hid(detid);
        layer = hid.depth();
        if (detid.subdetId() == HcalSubdetector::HcalOuter)
          layer += 1;
      }
      if (layer > maxLayerIndex_) {
        throw cms::Exception("BarrelSoARecHitsProducer")
            << "rechit " << detid.rawId() << " is on layer " << layer << ", above maxLayerIndex " << maxLayerIndex_;
      }
      return layer;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(BarrelSoARecHitsProducer);
