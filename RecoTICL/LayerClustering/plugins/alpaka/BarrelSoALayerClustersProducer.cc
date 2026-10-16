#include <string>
#include <utility>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/BarrelSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/BarrelSoARecHitsExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "LayerClustersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Builds the layer clusters of the ECAL or HCAL barrel (energy, number of cells, seed, energy-weighted position and
  // time) from its rechits and their clustering (the layer cluster and the seed flag of each rechit)
  class BarrelSoALayerClustersProducer : public stream::SynchronizingEDProducer<> {
  public:
    BarrelSoALayerClustersProducer(edm::ParameterSet const& config)
        : SynchronizingEDProducer(config),
          recHitsToken_{consumes(config.getParameter<edm::InputTag>("barrelRecHitsSoA"))},
          recHitsClusteringToken_{consumes(config.getParameter<edm::InputTag>("barrelRecHitsLayerClustersSoA"))},
          layerClustersToken_{produces()},
          algoId_(config.getParameter<std::string>("detector") == "EB" ? ::reco::CaloCluster::barrel_em
                                                                         : ::reco::CaloCluster::barrel_had),
          nHitsTime_(config.getParameter<unsigned int>("nHitsTime")) {}

    // The number of layer clusters is on the device, and the host needs it to allocate them
    void acquire(device::Event const& iEvent, device::EventSetup const&) override {
      auto const recHitsClustering = iEvent.get(recHitsClusteringToken_).view();
      alpaka::memcpy(iEvent.queue(),
                     cms::alpakatools::make_host_view<unsigned int>(nLayerClusters_),
                     cms::alpakatools::make_device_view<const unsigned int>(
                         alpaka::getDev(iEvent.queue()), recHitsClustering.numberOfClustersScalar()));
    }

    void produce(device::Event& iEvent, device::EventSetup const&) override {
      reco::CaloClusterDeviceCollection layerClusters(
          iEvent.queue(), nLayerClusters_, nLayerClusters_, nLayerClusters_, nLayerClusters_);
      // thresholdW0 and positionDeltaRho2 are only used by the HGCAL silicon
      algo_.run(iEvent.queue(),
                nLayerClusters_,
                0.f,
                0.f,
                LayerClustersGeometry::kBarrel,
                nHitsTime_,
                algoId_,
                iEvent.get(recHitsToken_).view(),
                iEvent.get(recHitsClusteringToken_).view(),
                layerClusters.view());
      iEvent.emplace(layerClustersToken_, std::move(layerClusters));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("barrelRecHitsLayerClustersSoA", edm::InputTag("TO BE DEFINED"));
      desc.add<edm::InputTag>("barrelRecHitsSoA", edm::InputTag("TO BE DEFINED"));
      desc.ifValue(edm::ParameterDescription<std::string>(
                       "detector", "EB", true, edm::Comment("the barrel calorimeter: EB (ECAL) or HB (HCAL)")),
                   edm::allowedValues<std::string>("EB", "HB"));
      desc.add<unsigned int>("nHitsTime", 3)
          ->setComment("Minimum number of rechits with a time for a layer cluster to get a time.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    device::EDGetToken<BarrelSoARecHitsDeviceCollection> const recHitsToken_;
    device::EDGetToken<BarrelSoARecHitsExtraDeviceCollection> const recHitsClusteringToken_;
    device::EDPutToken<reco::CaloClusterDeviceCollection> const layerClustersToken_;
    ::reco::CaloCluster::AlgoId const algoId_;
    unsigned int const nHitsTime_;
    LayerClustersSoAAlgoWrapper const algo_{};
    // copied from the device in acquire, used in produce
    unsigned int nLayerClusters_ = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(BarrelSoALayerClustersProducer);
