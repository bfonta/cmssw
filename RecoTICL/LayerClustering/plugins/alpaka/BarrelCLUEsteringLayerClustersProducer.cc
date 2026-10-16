#include <cstdint>
#include <span>
#include <vector>

#include "DataFormats/HGCalReco/interface/alpaka/BarrelSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/BarrelSoARecHitsExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "CLUEsteringAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Clusters the rechits of the ECAL or HCAL barrel, layer by layer, with CLUEstering in (eta, phi): produces the layer
  // cluster and the seed flag of each rechit, and the number of layer clusters
  class BarrelCLUEsteringLayerClustersProducer : public stream::EDProducer<> {
  public:
    BarrelCLUEsteringLayerClustersProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          recHitsToken_{consumes(config.getParameter<edm::InputTag>("barrelRecHitsSoA"))},
          layerSizesToken_{consumes<std::vector<uint32_t>>(
              edm::InputTag(config.getParameter<edm::InputTag>("barrelRecHitsSoA").label(), "layerSizes"))},
          recHitsClusteringToken_{produces()},
          deltac_(config.getParameter<float>("deltac")),
          kappa_(config.getParameter<float>("kappa")),
          outlierDelta_(config.getParameter<float>("outlierDelta")) {}

    void produce(device::Event& iEvent, device::EventSetup const&) override {
      auto const& recHits = iEvent.get(recHitsToken_);
      auto const& layerSizes = iEvent.get(layerSizesToken_);
      // One row per rechit, like the rechit SoA
      BarrelSoARecHitsExtraDeviceCollection recHitsClustering(iEvent.queue(), recHits->metadata().size());

      algo_.run(iEvent.queue(),
                recHits->metadata().size(),
                deltac_,
                kappa_,
                outlierDelta_,
                true,
                std::span<const uint32_t>(layerSizes),
                recHits.view(),
                recHitsClustering.view());
      iEvent.emplace(recHitsClusteringToken_, std::move(recHitsClustering));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("barrelRecHitsSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<float>("deltac", 1.8 * 0.0175)->setComment("Density radius, in (eta, phi).");
      desc.add<float>("kappa", 1.)->setComment("Minimum density of a seed, in units of the rechit noise.");
      // Unlike the HGCAL producer, which takes a factor of deltac (outlierDeltaFactor), the barrel takes an absolute
      // outlier distance, as the CPU barrel layer clustering (BarrelCLUEAlgoT) does with its outlierDeltaFactor
      desc.add<float>("outlierDelta", 2.7 * 0.0175)
          ->setComment("Absolute outlier distance, in (eta, phi), not a factor of deltac.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    device::EDGetToken<BarrelSoARecHitsDeviceCollection> const recHitsToken_;
    edm::EDGetTokenT<std::vector<uint32_t>> const layerSizesToken_;
    device::EDPutToken<BarrelSoARecHitsExtraDeviceCollection> const recHitsClusteringToken_;
    float const deltac_;
    float const kappa_;
    float const outlierDelta_;
    CLUEsteringAlgoWrapper const algo_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(BarrelCLUEsteringLayerClustersProducer);
