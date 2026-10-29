#ifndef RecoTICL_PatternRecognition_plugins_alpaka_HGCalTrackstersSoAAlgoWrapper_h
#define RecoTICL_PatternRecognition_plugins_alpaka_HGCalTrackstersSoAAlgoWrapper_h

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalTrackstersSoAAlgoWrapper {
  public:
    struct Parameters {
      // The layer clusters that count to the electromagnetic energy: in the barrel the ECAL ones, in HGCAL those with
      // |z| up to zLimitEm
      bool barrel;
      float zLimitEm;
      // Whether the PCA only uses the most energetic layer cluster of each layer near the most energetic one
      bool pcaCleaning;
    };

    // Fills the tracksters from the clustering of the layer clusters: the layer clusters of each trackster, and its
    // energy, barycenter, time and principal components, as ticl::assignPCAtoTracksters does with energy weighting
    // and the local time, except for the spreads. The sums over the layer clusters of a trackster accumulate in
    // their order, one thread per trackster, so the tracksters are the same on every backend and every run.
    void run(Queue& queue,
             const Parameters& parameters,
             reco::CaloClusterDeviceCollection::ConstView layerClusters,
             HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
             ticl::TracksterDeviceCollection::View tracksters) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_PatternRecognition_plugins_alpaka_HGCalTrackstersSoAAlgoWrapper_h
