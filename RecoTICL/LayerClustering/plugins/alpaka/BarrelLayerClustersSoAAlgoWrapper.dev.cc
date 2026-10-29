#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "BarrelLayerClustersSoAAlgoWrapper.h"
#include "ConstantsForClusters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;
  using namespace hgcal::constants;

  namespace {

    // The double-precision scratch sums of a cluster: energy and energy-weighted x, y, z
    constexpr uint32_t kSumsPerCluster = 4;

    // Energy, number of hits, seed and energy-weighted position of each cluster.
    // Without sharing every rechit belongs to at most one cluster, with fraction 1.
    // The sums are accumulated in double precision: the products of the two floats are exact in double, and the
    // sums are exact enough that their float rounding does not depend on the order of the additions, so the
    // layer clusters are the same on every backend.
    template <typename CAL>
    struct KernelAccumulate {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    reco::PFRecHitDeviceCollection::ConstView inputRecHits,
                                    HGCalSoARecHitsExtraDeviceCollection::ConstView inputClusters,
                                    double* sums,
                                    reco::CaloClusterDeviceCollection::View outputs) const {
        for (auto i : cms::alpakatools::uniform_elements(acc, inputClusters.metadata().size())) {
          const auto clIdx = inputClusters[i].clusterIndex();
          if (clIdx == kInvalidCluster)
            continue;
          const auto energy = inputRecHits[i].energy();
          sums[kSumsPerCluster * clIdx] += static_cast<double>(energy);
          sums[kSumsPerCluster * clIdx + 1] += static_cast<double>(inputRecHits[i].x()) * energy;
          sums[kSumsPerCluster * clIdx + 2] += static_cast<double>(inputRecHits[i].y()) * energy;
          sums[kSumsPerCluster * clIdx + 3] += static_cast<double>(inputRecHits[i].z()) * energy;
          outputs.position()[clIdx].cells() += 1;
          if (inputClusters[i].isSeed() == 1) {
            outputs.indexes()[clIdx].seedID() = inputRecHits[i].detId();
            outputs.position()[clIdx].layer() = CAL::layer(inputRecHits, i);
          }
        }
      }
    };

    struct KernelFinalise {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const double* sums,
                                    reco::CaloClusterDeviceCollection::View outputs,
                                    uint32_t numberOfClusters) const {
        for (auto clIdx : cms::alpakatools::uniform_elements(acc, numberOfClusters)) {
          // As in the legacy BarrelCLUEAlgo, a cluster without energy is placed at the origin
          const auto energy = static_cast<float>(sums[kSumsPerCluster * clIdx]);
          const auto invEnergy = (energy != 0.f) ? 1.f / energy : 0.f;
          outputs.energy()[clIdx].energy() = energy;
          outputs.position()[clIdx].x() = static_cast<float>(sums[kSumsPerCluster * clIdx + 1]) * invEnergy;
          outputs.position()[clIdx].y() = static_cast<float>(sums[kSumsPerCluster * clIdx + 2]) * invEnergy;
          outputs.position()[clIdx].z() = static_cast<float>(sums[kSumsPerCluster * clIdx + 3]) * invEnergy;
          // Same defaults as reco::CaloCluster; the time is computed when converting to the legacy format
          outputs.energy()[clIdx].correctedEnergy() = -1.f;
          outputs.energy()[clIdx].correctedEnergyUncertainty() = -1.f;
          outputs.timing()[clIdx].time() = -99.f;
          outputs.timing()[clIdx].timeError() = -1.f;
          outputs.indexes()[clIdx].flags() = 0;
        }
      }
    };

  }  // namespace

  template <typename CAL>
  void BarrelLayerClustersSoAAlgoWrapper<CAL>::run(Queue& queue,
                                                   unsigned int numberOfClusters,
                                                   reco::PFRecHitDeviceCollection::ConstView inputRecHits,
                                                   HGCalSoARecHitsExtraDeviceCollection::ConstView inputClusters,
                                                   reco::CaloClusterDeviceCollection::View outputs) const {
    if (numberOfClusters == 0)
      return;

    // The double-precision accumulators of the cluster sums: energy and energy-weighted position
    auto sums = make_device_buffer<double[]>(queue, kSumsPerCluster * numberOfClusters);
    alpaka::memset(queue, sums, 0x0);
    auto cells = make_device_view<int>(queue, outputs.position().cells(), numberOfClusters);
    alpaka::memset(queue, cells, 0x0);

    const auto items = 256u;

    // single thread: the sums of a cluster then accumulate in the order of the input rechits on every backend (a
    // parallel loop would order the additions by thread scheduling)
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, 1), KernelAccumulate<CAL>{}, inputRecHits, inputClusters, sums.data(), outputs);

    const auto clustersWorkDiv = make_workdiv<Acc1D>(divide_up_by(numberOfClusters, items), items);
    alpaka::exec<Acc1D>(queue, clustersWorkDiv, KernelFinalise{}, sums.data(), outputs, numberOfClusters);
  }

  template class BarrelLayerClustersSoAAlgoWrapper<barrelclue::ECAL>;
  template class BarrelLayerClustersSoAAlgoWrapper<barrelclue::HCAL>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
