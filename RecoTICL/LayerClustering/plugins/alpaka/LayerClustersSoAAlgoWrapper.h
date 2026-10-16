#ifndef RecoTICL_LayerClustering_plugins_alpaka_LayerClustersSoAAlgoWrapper_h
#define RecoTICL_LayerClustering_plugins_alpaka_LayerClustersSoAAlgoWrapper_h

#include <cstdint>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // What the rechit coordinates (dim1, dim2, dim3) are, and how the layer cluster position and time are computed
  enum class LayerClustersGeometry : uint8_t {
    // HGCAL silicon: (x, y, z). Position from the log-weighted rechits around the most energetic one, with a time.
    kSilicon,
    // HGCAL scintillator: (eta, phi, z). Energy-weighted position, without a time.
    kScintillator,
    // ECAL and HCAL barrel: (eta, phi, r), r being the distance from the origin. Energy-weighted position (x, y and
    // z), with a time, like BarrelCLUEAlgoT.
    kBarrel
  };

  // Shared by the HGCAL and barrel layer clustering (the barrel rechit SoAs are aliases of the HGCAL ones)
  class LayerClustersSoAAlgoWrapper {
  public:
    // Fills the nLayerClusters layer clusters (energy, number of cells, seed, position and time) from the rechits and
    // their clustering (the layer cluster and the seed flag of each rechit). A layer cluster needs at least nHitsTime
    // rechits with a time to get a time. thresholdW0 and positionDeltaRho2 are only used for kSilicon.
    void run(Queue& queue,
             uint32_t nLayerClusters,
             float thresholdW0,
             float positionDeltaRho2,
             LayerClustersGeometry geometry,
             uint32_t nHitsTime,
             ::reco::CaloCluster::AlgoId algoId,
             HGCalSoARecHitsDeviceCollection::ConstView recHits,
             HGCalSoARecHitsExtraDeviceCollection::ConstView recHitsClustering,
             reco::CaloClusterDeviceCollection::View layerClusters) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_LayerClustering_plugins_alpaka_LayerClustersSoAAlgoWrapper_h
