#ifndef RecoTICL_PatternRecognition_plugins_alpaka_BarrelCLUEsteringTrackstersAlgoWrapper_h
#define RecoTICL_PatternRecognition_plugins_alpaka_BarrelCLUEsteringTrackstersAlgoWrapper_h

#include <array>
#include <cstdint>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BarrelCLUEsteringTrackstersAlgoWrapper {
  public:
    // The barrel calorimeters: ECAL (EB) and HCAL (HB), in the order of DetId::Ecal, DetId::Hcal
    static constexpr uint32_t kComponents = 2;

    struct Parameters {
      // The clustering distances, in units of sigmaT transversely and of layerScale layers longitudinally
      float deltac;
      float outlierDistance;
      float seedingDistance;

      // The minimum energy density of a seed [GeV] in each barrel calorimeter
      std::array<float, kComponents> rhoc;

      // The transverse (eta, phi) scale of each barrel calorimeter
      std::array<float, kComponents> sigmaT;
      float layerScale;

      // Whether the layer clusters of the same layer are neighbours. The legacy barrel CLUE3D does not let them
      // contribute to the density, nor be the nearest higher.
      bool sameLayerNeighbours;

      // The 3D clusters with fewer layer clusters are not tracksters
      uint32_t minLayerClusters;
    };

    // Clusters the barrel layer clusters into tracksters with CLUEstering: writes the trackster of each layer
    // cluster, the number of tracksters and the number of layer clusters in them.
    void run(Queue& queue,
             const Parameters& parameters,
             reco::CaloClusterDeviceCollection::ConstView layerClusters,
             HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_PatternRecognition_plugins_alpaka_BarrelCLUEsteringTrackstersAlgoWrapper_h
