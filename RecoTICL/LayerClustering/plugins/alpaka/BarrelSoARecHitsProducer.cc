#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/HcalObjects/interface/HcalPFCuts.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BarrelSoARecHitsProducer : public stream::EDProducer<> {
  public:
    BarrelSoARecHitsProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          detector_(config.getParameter<std::string>("detector")),
          hitsToken_(consumes<reco::PFRecHitCollection>(config.getParameter<edm::InputTag>("recHits"))),
          ebThresholdsToken_(
              consumesCollector().esConsumes<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd>(edm::ESInputTag("", ""))),
          hcalThresholdsToken_(
              consumesCollector().esConsumes<HcalPFCuts, HcalPFCutsRcd>(edm::ESInputTag("", ""))),
          deviceToken_{produces()} {
      if (detector_ != "EB" && detector_ != "HB") {
        throw cms::Exception("Configuration") << "BarrelSoARecHitsProducer: detector must be either 'EB' or 'HB', got '"
                                              << detector_ << "'";
      }
    }

    ~BarrelSoARecHitsProducer() override = default;

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto const& hits = iEvent.get(hitsToken_);
      auto const& ebThresholds = iSetup.getData(ebThresholdsToken_);
      auto const& hcalThresholds = iSetup.getData(hcalThresholdsToken_);

      HGCalSoARecHitsHostCollection cells(iEvent.queue(), hits.size());
      auto cellsView = cells.view();

      for (unsigned int i = 0; i < hits.size(); ++i) {
        auto const& hit = hits[i];
        DetId const detid(hit.detId());
        auto const& position = hit.positionREP();

        float sigmaNoise = 1.f;
        int layer = 0;
        if (detector_ == "EB") {
          sigmaNoise = ebThresholds[detid];
          layer = 0;
        } else {
          if (detid.det() == DetId::Hcal) {
            HcalDetId const hid(detid);
            layer = hid.depth();
            if (detid.subdetId() == HcalSubdetector::HcalOuter) {
              ++layer;
            }
            const auto* item = hcalThresholds.getValues(detid);
            if (item != nullptr) {
              sigmaNoise = item->seedThreshold();
            }
          }
        }

        auto entry = cellsView[i];
        entry.dim1() = position.eta();
        entry.dim2() = position.phi();
        entry.dim3() = position.rho();
        entry.energy() = hit.energy();
        entry.mipEnergy() = hit.energy();
        entry.sigmaNoise() = sigmaNoise;
        entry.layer() = layer;
        entry.recHitIndex() = i;
        entry.detid() = detid.rawId();
        entry.time() = hit.time();
        entry.timeError() = -1.f;
      }

      if constexpr (!std::is_same_v<Device, alpaka_common::DevHost>) {
        HGCalSoARecHitsDeviceCollection deviceProduct{iEvent.queue(), cells->metadata().size()};
        alpaka::memcpy(iEvent.queue(), deviceProduct.buffer(), cells.const_buffer());
        iEvent.emplace(deviceToken_, std::move(deviceProduct));
      } else {
        iEvent.emplace(deviceToken_, std::move(cells));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("detector", "EB")->setComment("Barrel detector to convert to SoA: EB or HB");
      desc.add<edm::InputTag>("recHits", edm::InputTag("particleFlowRecHitECAL"));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    std::string detector_;
    edm::EDGetTokenT<reco::PFRecHitCollection> const hitsToken_;
    edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> const ebThresholdsToken_;
    edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> const hcalThresholdsToken_;
    device::EDPutToken<HGCalSoARecHitsDeviceCollection> const deviceToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(BarrelSoARecHitsProducer);
