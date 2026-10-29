#include <algorithm>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "BarrelCLUEsteringTrackstersAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Clusters the layer clusters of the barrel calorimeters (EB and HB) together into tracksters with CLUEstering:
  // produces the trackster of each layer cluster, the number of tracksters and the number of layer clusters in them.
  // The clustering coordinates are (eta / sigmaT, phi / sigmaT), with phi periodic, and the distance is
  // max(transverse distance / mean sigmaT of the pair, layer distance / layerScale).
  class BarrelCLUEsteringTrackstersProducer : public stream::EDProducer<> {
  public:
    BarrelCLUEsteringTrackstersProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("layerClustersSoA"))},
          layerClustersClusteringToken_{produces()},
          parameters_{makeParameters(config)} {}

    void produce(device::Event& iEvent, const device::EventSetup&) override {
      if (not logged_) {
        logged_ = true;
        // the tuning in the log, once per stream: a wrong trackster number is otherwise hard to trace back to the
        // parameters used
        edm::LogPrint("BarrelCLUEsteringTrackstersProducer")
            << "clustering with deltac " << parameters_.deltac << ", outlierDistance " << parameters_.outlierDistance
            << ", seedingDistance " << parameters_.seedingDistance << ", rhoc " << parameters_.rhoc[0] << '/'
            << parameters_.rhoc[1] << " GeV, sigmaT " << parameters_.sigmaT[0] << '/' << parameters_.sigmaT[1]
            << ", layerScale " << parameters_.layerScale << ", sameLayerNeighbours " << parameters_.sameLayerNeighbours
            << ", minLayerClusters " << parameters_.minLayerClusters;
      }
      const auto layerClusters = iEvent.get(layerClustersToken_).const_view();

      HGCalSoALayerClustersExtraDeviceCollection layerClustersClustering(iEvent.queue(),
                                                                         layerClusters.position().metadata().size());

      algo_.run(iEvent.queue(), parameters_, layerClusters, layerClustersClustering.view());
      iEvent.emplace(layerClustersClusteringToken_, std::move(layerClustersClustering));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      // The defaults mirror the legacy barrel CLUE3D tuning of CLUE3DBarrel_cff.py: criticalDensity 0.5 GeV,
      // criticalEtaPhiDistance 3 sigmaT, outlierMultiplier 2, no same-layer neighbours, minNumLayerCluster 1.
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("layerClustersSoA", edm::InputTag("barrelSoAMergeLayerClusters"))
          ->setComment("The EB and HB layer clusters, merged on the device.");
      desc.add<float>("deltac", 1.5)->setComment("Density radius, in the clustering coordinates.");
      desc.add<float>("outlierDistance", 6.)->setComment("Maximum distance to the nearest higher.");
      desc.add<float>("seedingDistance", 3.)
          ->setComment("Maximum distance to the nearest higher of the points above the seed threshold.");
      desc.add<std::vector<double>>("rhoc", {0.5, 0.5})
          ->setComment(
              "Minimum energy density of a seed [GeV] of EB, HB. The points below it use outlierDistance for their "
              "nearest higher, those above seedingDistance.");
      desc.add<std::vector<double>>("sigmaT", {0.0175, 0.087})
          ->setComment("Transverse (eta, phi) scale of EB and HB: the size of their cells.");
      desc.add<float>("layerScale", 4.)->setComment("Longitudinal scale, in layers (EB is layer 0, HB its depths).");
      desc.add<bool>("sameLayerNeighbours", false)
          ->setComment(
              "Whether the layer clusters of the same layer contribute to the density of each other and can be the "
              "nearest higher of each other. They cannot in the legacy barrel CLUE3D.");
      desc.add<unsigned int>("minLayerClusters", 1)->setComment("Minimum number of layer clusters of a trackster.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    static BarrelCLUEsteringTrackstersAlgoWrapper::Parameters makeParameters(const edm::ParameterSet& config) {
      BarrelCLUEsteringTrackstersAlgoWrapper::Parameters parameters{
          .deltac = config.getParameter<float>("deltac"),
          .outlierDistance = config.getParameter<float>("outlierDistance"),
          .seedingDistance = config.getParameter<float>("seedingDistance"),
          .rhoc = {},
          .sigmaT = {},
          .layerScale = config.getParameter<float>("layerScale"),
          .sameLayerNeighbours = config.getParameter<bool>("sameLayerNeighbours"),
          .minLayerClusters = config.getParameter<unsigned int>("minLayerClusters")};

      // one value per barrel calorimeter
      auto copyPerComponent = [&config](const char* name, auto& destination) {
        const auto& values = config.getParameter<std::vector<double>>(name);
        if (values.size() != destination.size()) {
          throw edm::Exception(edm::errors::Configuration)
              << "BarrelCLUEsteringTrackstersProducer: '" << name << "' needs one value per barrel calorimeter (EB, HB).";
        }
        std::ranges::copy(values, destination.begin());
      };
      copyPerComponent("rhoc", parameters.rhoc);
      copyPerComponent("sigmaT", parameters.sigmaT);
      if (std::ranges::any_of(parameters.rhoc, [](float rhoc) { return not(rhoc > 0.f); })) {
        throw edm::Exception(edm::errors::Configuration)
            << "BarrelCLUEsteringTrackstersProducer: 'rhoc' must be positive, as the seed threshold of each barrel "
               "calorimeter.";
      }
      if (std::ranges::any_of(parameters.sigmaT, [](float sigmaT) { return not(sigmaT > 0.f); })) {
        throw edm::Exception(edm::errors::Configuration)
            << "BarrelCLUEsteringTrackstersProducer: 'sigmaT' must be positive, as the transverse scale of each "
               "barrel calorimeter.";
      }
      if (not(parameters.deltac > 0.f) or not(parameters.layerScale > 0.f)) {
        throw edm::Exception(edm::errors::Configuration)
            << "BarrelCLUEsteringTrackstersProducer: 'deltac' and 'layerScale' must be positive, as the density "
               "radius and the longitudinal scale.";
      }
      // CLUEstering sizes the nearest-higher search window with the outlier distance: a larger seeding distance
      // would look for nearest higher points outside of the searched tiles
      if (parameters.seedingDistance > parameters.outlierDistance) {
        throw edm::Exception(edm::errors::Configuration)
            << "BarrelCLUEsteringTrackstersProducer: 'seedingDistance' must not exceed 'outlierDistance', which "
               "sizes the nearest-higher search window.";
      }
      if (parameters.minLayerClusters < 1) {
        throw edm::Exception(edm::errors::Configuration)
            << "BarrelCLUEsteringTrackstersProducer: 'minLayerClusters' must be at least 1, as the minimum number of "
               "layer clusters of a trackster.";
      }
      return parameters;
    }

    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    const device::EDPutToken<HGCalSoALayerClustersExtraDeviceCollection> layerClustersClusteringToken_;
    const BarrelCLUEsteringTrackstersAlgoWrapper::Parameters parameters_;
    const BarrelCLUEsteringTrackstersAlgoWrapper algo_{};
    bool logged_ = false;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(BarrelCLUEsteringTrackstersProducer);
