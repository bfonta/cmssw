#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <span>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "BarrelCLUEsteringAlgoWrapper.h"
#include "ConstantsForClusters.h"

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace hgcal::constants;

  namespace {

    // The PFRecHit SoA is allocated with the capacity of the input CaloRecHits: the kernels run on the
    // whole capacity and skip the entries beyond the number of PFRecHits, which is only known on the device.
    template <typename CAL>
    struct KernelCountHitsPerLayer {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    reco::PFRecHitDeviceCollection::ConstView hits,
                                    uint32_t* layerSizes) const {
        const auto size = hits.size();
        for (auto i : uniform_elements(acc, hits.metadata().size())) {
          if (i >= size)
            continue;
          const uint32_t layer = CAL::layer(hits, i);
          if (layer < CAL::kNumberOfLayers)
            alpaka::atomicAdd(acc, &layerSizes[layer], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    };

    // Kernel to mark the rechits of one layer: the inclusive scan of the marks is the rank of each rechit within its
    // layer, which packs the points in the order of the input rechits on every backend
    template <typename CAL>
    struct KernelLayerFlags {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    reco::PFRecHitDeviceCollection::ConstView hits,
                                    uint32_t wantedLayer,
                                    uint32_t* flags,
                                    uint32_t size) const {
        for (auto i : uniform_elements(acc, size)) {
          flags[i] = (CAL::layer(hits, i) == wantedLayer) ? 1u : 0u;
        }
      }
    };

    // Kernel to turn the inclusive scan of the marks into the rank within the layer
    struct KernelRankWithinLayer {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const uint32_t* flags,
                                    const uint32_t* flagsEnd,
                                    uint32_t* rank,
                                    uint32_t size) const {
        for (auto i : uniform_elements(acc, size)) {
          if (flags[i] != 0) {
            rank[i] = flagsEnd[i] - 1;
          }
        }
      }
    };

    template <typename CAL>
    struct KernelPrepareInputs {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    reco::PFRecHitDeviceCollection::ConstView hits,
                                    typename CAL::NoiseType::ConstView noise,
                                    const uint32_t* layerCursors,
                                    const uint32_t* rank,
                                    BarrelCLUEsteringInputsDeviceCollection::View clueInputs,
                                    uint32_t size) const {
        for (auto i : uniform_elements(acc, size)) {
          const uint32_t layer = CAL::layer(hits, i);
          if (layer >= CAL::kNumberOfLayers)
            continue;
          // The rank of the rechit within its layer, in input order: the cluster sums of CLUEstering would otherwise
          // round in an order that depends on the thread scheduling, which flips the clustering decisions of the
          // tie-rich barrel rechits
          const auto j = layerCursors[layer] + rank[i];
          // The coordinates are computed in double precision and rounded to float once: the float store removes
          // the 1-ULP differences between the transcendentals of the backends
          const double x = hits[i].x();
          const double y = hits[i].y();
          const double z = hits[i].z();
          clueInputs[j].eta() = std::asinh(z / std::sqrt(x * x + y * y));
          clueInputs[j].phi() = std::atan2(y, x);
          clueInputs[j].energy() = hits[i].energy();
          clueInputs[j].sigmaNoise() = CAL::sigmaNoise(noise, hits[i].denseId());
          clueInputs[j].detId() = hits[i].detId();
          clueInputs[j].hitIndex() = i;
        }
      }
    };

    // The clustering distance of the 2D clustering: the Euclidean distance in (eta, phi) with phi periodic, like
    // clue::PeriodicEuclideanMetric<2>. It is computed in double precision and rounded to float once, like the
    // coordinates: CLUEstering then compares the same values on every backend, and its distance ties break by tag
    // deterministically.
    struct DeterministicPeriodicEuclideanMetric {
      using value_type = float;

      float phiPeriod;

      template <typename TPoints>
      ALPAKA_FN_HOST_ACC float operator()(const TPoints& points, std::size_t i, std::size_t j) const {
        const auto pi = points[static_cast<int>(i)];
        const auto pj = points[static_cast<int>(j)];
        const double dEta = pi[0] - pj[0];
        const double dPhi = std::abs(pi[1] - pj[1]);
        const double dPhiWrapped = std::min(dPhi, static_cast<double>(phiPeriod) - dPhi);
        return static_cast<float>(std::sqrt(dEta * dEta + dPhiWrapped * dPhiWrapped));
      }
    };

    // The cluster indices assigned by CLUEstering follow the order in which it finds the seeds, which is an internal
    // detail of the algorithm: renumber the clusters ordering them by the (layer, detId) of their seed, a canonical
    // order independent of that numbering and of the backend.
    template <typename CAL>
    struct KernelRankClusters {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const int32_t* seeds,
                                    reco::PFRecHitDeviceCollection::ConstView hits,
                                    BarrelCLUEsteringInputsDeviceCollection::ConstView clueInputs,
                                    int32_t* newClusterIndex,
                                    uint32_t nseeds) const {
        auto seedKey = [&](uint32_t k) -> uint64_t {
          const auto& seed = clueInputs[seeds[k]];
          return (static_cast<uint64_t>(CAL::layer(hits, seed.hitIndex())) << 32) | seed.detId();
        };
        for (auto k : uniform_elements(acc, nseeds)) {
          const uint64_t key = seedKey(k);
          int32_t rank = 0;
          for (uint32_t m = 0; m < nseeds; ++m) {
            rank += (seedKey(m) < key) ? 1 : 0;
          }
          newClusterIndex[clueInputs[seeds[k]].clusterIndex()] = rank;
        }
      }
    };

    // Bring the cluster indices back to the order of the PFRecHits
    struct KernelSetClusterIndex {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    BarrelCLUEsteringInputsDeviceCollection::ConstView clueInputs,
                                    const int32_t* newClusterIndex,
                                    HGCalSoARecHitsExtraDeviceCollection::View outputs) const {
        for (auto j : uniform_elements(acc, clueInputs.metadata().size())) {
          const auto clusterIndex = clueInputs[j].clusterIndex();
          outputs[clueInputs[j].hitIndex()].clusterIndex() =
              (clusterIndex == kInvalidCluster) ? kInvalidCluster : newClusterIndex[clusterIndex];
        }
      }
    };

    struct KernelSetSeeds {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const int32_t* seeds,
                                    BarrelCLUEsteringInputsDeviceCollection::ConstView clueInputs,
                                    HGCalSoARecHitsExtraDeviceCollection::View outputs,
                                    uint32_t nseeds) const {
        for (auto k : uniform_elements(acc, nseeds)) {
          outputs[clueInputs[seeds[k]].hitIndex()].isSeed() = 1;
        }
      }
    };

  }  // namespace

  template <typename CAL>
  void BarrelCLUEsteringAlgoWrapper<CAL>::countHitsPerLayer(Queue& queue,
                                                            reco::PFRecHitDeviceCollection::ConstView inputs,
                                                            std::span<uint32_t> layerSizes) const {
    auto d_layerSizes = make_device_buffer<uint32_t[]>(queue, CAL::kNumberOfLayers);
    alpaka::memset(queue, d_layerSizes, 0x0);
    const uint32_t capacity = inputs.metadata().size();
    if (capacity > 0) {
      const auto items = 256u;
      const auto workDiv = make_workdiv<Acc1D>(divide_up_by(capacity, items), items);
      alpaka::exec<Acc1D>(queue, workDiv, KernelCountHitsPerLayer<CAL>{}, inputs, d_layerSizes.data());
    }
    alpaka::memcpy(queue, make_host_view(layerSizes), d_layerSizes);
  }

  template <typename CAL>
  void BarrelCLUEsteringAlgoWrapper<CAL>::run(Queue& queue,
                                              uint32_t size,
                                              float dc,
                                              float kappa,
                                              std::span<const uint32_t> layerSizes,
                                              const reco::PFRecHitDeviceCollection::ConstView inputs,
                                              const typename CAL::NoiseType::ConstView noise,
                                              HGCalSoARecHitsExtraDeviceCollection::View outputs) const {
    auto nClusters = make_device_view<unsigned int>(queue, outputs.numberOfClustersScalar());
    alpaka::memset(queue, nClusters, 0x0);
    if (size == 0)
      return;

    // Rechits on a layer outside [0, CAL::kNumberOfLayers) are not counted and stay unclustered
    const auto nClusteredHits = std::accumulate(layerSizes.begin(), layerSizes.end(), 0u);

    auto clusterIndexView = make_device_view(queue, outputs.clusterIndex().data(), size);
    alpaka::memset(queue, clusterIndexView, kInvalidClusterByte);
    auto isSeedView = make_device_view(queue, outputs.isSeed().data(), size);
    alpaka::memset(queue, isSeedView, 0x0);
    if (nClusteredHits == 0)
      return;

    // Each layer starts at the sum of the sizes of the previous ones
    auto h_layerCursors = make_host_buffer<uint32_t[]>(queue, CAL::kNumberOfLayers);
    std::exclusive_scan(layerSizes.begin(), layerSizes.end(), h_layerCursors.data(), 0u);
    auto d_layerCursors = make_device_buffer<uint32_t[]>(queue, CAL::kNumberOfLayers);
    alpaka::memcpy(queue, d_layerCursors, h_layerCursors);

    const auto items = 256u;

    // The rank of every rechit within its layer, in the order of the input rechits: the points of a layer are then
    // packed deterministically on every backend (an atomic counter would order them by thread scheduling)
    auto rank = make_device_buffer<uint32_t[]>(queue, size);
    auto flags = make_device_buffer<uint32_t[]>(queue, size);
    auto flagsEnd = make_device_buffer<uint32_t[]>(queue, size);
    const auto fillWorkDiv = make_workdiv<Acc1D>(divide_up_by(size, items), items);
    for (uint32_t layer = 0; layer < CAL::kNumberOfLayers; ++layer) {
      alpaka::exec<Acc1D>(queue, fillWorkDiv, KernelLayerFlags<CAL>{}, inputs, layer, flags.data(), size);
      iterativePrefixScan<Acc1D>(flags.data(), flagsEnd.data(), size, queue);
      alpaka::exec<Acc1D>(queue, fillWorkDiv, KernelRankWithinLayer{}, flags.data(), flagsEnd.data(), rank.data(), size);
    }

    BarrelCLUEsteringInputsDeviceCollection clueInputs(queue, nClusteredHits);
    auto clueInputs_v = clueInputs.view();
    alpaka::exec<Acc1D>(queue,
                        fillWorkDiv,
                        KernelPrepareInputs<CAL>{},
                        inputs,
                        noise,
                        d_layerCursors.data(),
                        rank.data(),
                        clueInputs_v,
                        size);

    clue::ConstPointsDevice<2, float> d_points(queue,
                                               static_cast<int32_t>(nClusteredHits),
                                               clueInputs_v.eta().data(),
                                               clueInputs_v.phi().data(),
                                               clueInputs_v.energy().data(),
                                               clueInputs_v.clusterIndex().data());
    d_points.set_density_uncertainty(clueInputs_v.sigmaNoise());
    d_points.set_tags(clueInputs_v.detId());

    // As in the legacy BarrelCLUEAlgo, rechits below the density threshold are only attached to a
    // higher-density rechit within dc: the outlier distance is dc as well.
    clue::Clusterer<2> algo(queue, dc, kappa, dc);
    // Clustering in (eta, phi), with phi periodic
    constexpr float kTwoPi = 2.f * std::numbers::pi_v<float>;
    algo.setWrappedCoordinates(0, 1);
    // The same weight of the legacy BarrelCLUEAlgo: 1 for the rechit itself, 0.5 for its neighbours
    algo.make_clusters(queue,
                       d_points,
                       layerSizes,
                       DeterministicPeriodicEuclideanMetric{.phiPeriod = kTwoPi},
                       clue::FlatKernel<float>{0.5f});
    alpaka::wait(queue);

    const auto h_nClusters = static_cast<unsigned int>(d_points.n_clusters());
    auto seeds = algo.getSeeds();
    const auto nseeds = static_cast<uint32_t>(seeds.size());
    // Every cluster has exactly one seed
    assert(nseeds == h_nClusters);
    if (nseeds == 0)
      return;

    alpaka::memcpy(queue, nClusters, make_host_view(h_nClusters));

    auto newClusterIndex = make_device_buffer<int32_t[]>(queue, nseeds);
    const auto seedWorkDiv = make_workdiv<Acc1D>(divide_up_by(nseeds, items), items);
    alpaka::exec<Acc1D>(queue,
                        seedWorkDiv,
                        KernelRankClusters<CAL>{},
                        seeds.data(),
                        inputs,
                        clueInputs.const_view(),
                        newClusterIndex.data(),
                        nseeds);
    const auto clusterIndexWorkDiv = make_workdiv<Acc1D>(divide_up_by(nClusteredHits, items), items);
    alpaka::exec<Acc1D>(queue,
                        clusterIndexWorkDiv,
                        KernelSetClusterIndex{},
                        clueInputs.const_view(),
                        newClusterIndex.data(),
                        outputs);
    alpaka::exec<Acc1D>(
        queue, seedWorkDiv, KernelSetSeeds{}, seeds.data(), clueInputs.const_view(), outputs, nseeds);
    // The seeds are owned by the Clusterer, which goes out of scope at the end of this function
    alpaka::wait(queue);
  }

  template class BarrelCLUEsteringAlgoWrapper<barrelclue::ECAL>;
  template class BarrelCLUEsteringAlgoWrapper<barrelclue::HCAL>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
