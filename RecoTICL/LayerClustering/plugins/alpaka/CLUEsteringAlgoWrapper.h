#ifndef RecoTICL_LayerClustering_plugins_alpaka_CLUEsteringAlgoWrapper_h
#define RecoTICL_LayerClustering_plugins_alpaka_CLUEsteringAlgoWrapper_h

#include <cstdint>
#include <span>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Shared by the HGCAL and barrel layer clustering (the barrel rechit SoAs are aliases of the HGCAL ones)
  class CLUEsteringAlgoWrapper {
  public:
    // Clusters the rechits of each layer (one batch per layer, of batchItemSizes rechits) with CLUEstering: writes
    // the layer cluster and the seed flag of each rechit, and the number of layer clusters. With etaPhi, the rechits
    // cluster in (dim1, dim2) = (eta, phi), phi in [0, 2pi) and periodic, with a flat density kernel. outlierDelta is
    // the distance, in the clustering coordinates, beyond which a rechit without a nearest higher is an outlier.
    void run(Queue& queue,
             uint32_t nRecHits,
             float dc,
             float kappa,
             float outlierDelta,
             bool etaPhi,
             std::span<const uint32_t> batchItemSizes,
             HGCalSoARecHitsDeviceCollection::ConstView recHits,
             HGCalSoARecHitsExtraDeviceCollection::View recHitsClustering) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_LayerClustering_plugins_alpaka_CLUEsteringAlgoWrapper_h
