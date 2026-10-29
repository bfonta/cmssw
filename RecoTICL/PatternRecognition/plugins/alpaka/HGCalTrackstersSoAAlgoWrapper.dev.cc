// Eigen must precede the SoA headers to enable the Eigen columns of the TracksterSoA
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <bit>
#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "HGCalTrackstersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  namespace {

    constexpr int kInvalidTrackster = -1;

    // The PCA uses the most energetic layer cluster of each layer within kCleaningLayers layers of the most energetic
    // layer cluster of the trackster
    constexpr int32_t kCleaningLayers = 10;
    constexpr int32_t kCleaningWindow = 2 * kCleaningLayers + 1;
    // A layer cluster contributes to the time of its trackster if it is within this squared distance [cm^2] of the
    // line through the origin and the barycenter
    constexpr float kTimeMaxDistance2 = 9.f;
    constexpr float kSpeedOfLight = 29.9792458f;  // cm/ns
    // The time of a trackster without layer clusters with a time, and its error [ns]
    constexpr float kNoTime = -99.f;
    constexpr float kNoTimeError = -1.f;

    // The most energetic layer cluster of a set is found with one atomicMax on this key. The bits of a non-negative
    // float, read as an integer, are ordered as its value is. For equal energies, ~layerCluster makes the lowest
    // layer cluster index win. This assumes energy is always positive
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint64_t energyKey(const float energy, const uint32_t layerCluster) {
      return (static_cast<uint64_t>(std::bit_cast<uint32_t>(energy)) << 32) | ~layerCluster;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t layerClusterOfEnergyKey(const uint64_t key) {
      return ~static_cast<uint32_t>(key);
    }

    // The squared distance of a point from the line through the origin along direction. As in the legacy code, the
    // squares are summed in double precision (std::pow).
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float distance2FromLine(const float point[3], const float direction[3]) {
      float dot = 0.f;
      float direction2 = 0.f;
      for (int k = 0; k < 3; ++k) {
        dot += point[k] * direction[k];
        direction2 += direction[k] * direction[k];
      }
      const float projection = direction2 != 0.f ? dot / direction2 : 0.f;
      float distance2 = 0.f;
      for (int k = 0; k < 3; ++k) {
        const float d = point[k] - projection * direction[k];
        distance2 = static_cast<float>(distance2 + static_cast<double>(d) * d);
      }
      return distance2;
    }

    // The sums over the layer clusters of the trackster processed by a block, in shared memory. They are in double
    // precision: the products of the float values are exact in double, so the sums are exact enough that their float
    // rounding does not depend on the order of the atomic additions, and the tracksters are the same on every backend.
    struct TracksterSums {
      // all the layer clusters
      double energy;
      double emEnergy;
      double position[3];  // weighted by energy
      uint64_t maxEnergyKey;
      double time;  // weighted by 1 / timeError^2
      double timeWeight;
      // the layer clusters used by the PCA: the most energetic one of each layer of the cleaning window
      uint64_t layerMaxEnergyKey[kCleaningWindow];
      double cleanedEnergy;
      double cleanedPosition[3];  // weighted by energy
      // weighted by energy / cleanedEnergy
      double weight2;
      double covariance[6];  // xx, yx, yy, zx, zy, zz
      double sigmasPCA2[3];  // along the principal axes, in increasing eigenvalue order
      // the eigen-decomposition of the covariance matrix, in increasing eigenvalue order, the eigenvectors column-major
      double eigenvalues[3];
      double eigenvectors[9];
    };

    // Per layer cluster -> per trackster: the number of layer clusters
    struct KernelTracksterSizes {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
                                    uint32_t* tracksterSizes) const {
        for (auto i : uniform_elements(acc, layerClustersClustering.metadata().size())) {
          const int trackster = layerClustersClustering[i].tracksterIndex();
          if (trackster != kInvalidTrackster) {
            alpaka::atomicAdd(acc, &tracksterSizes[trackster], 1u);
          }
        }
      }
    };

    // Per layer cluster: its index in the layer clusters of its trackster, which end at the offset of the next
    // trackster. The sizes count down the layer clusters still to place, so that they fill the trackster from its start.
    struct KernelTracksterVertices {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
                                    uint32_t* tracksterSizes,
                                    ticl::TracksterDeviceCollection::View tracksters) const {
        auto vertices = tracksters.vertices();
        for (auto i : uniform_elements(acc, layerClustersClustering.metadata().size())) {
          const int trackster = layerClustersClustering[i].tracksterIndex();
          if (trackster == kInvalidTrackster) {
            continue;
          }
          const uint32_t vertex = vertices.offsets()[trackster + 1].keys_offsets() - tracksterSizes[trackster]--;
          vertices.content()[vertex].values() = i;
        }
      }
    };

    // Per trackster, one block: its energy, barycenter, time and principal components from its layer clusters, as
    // ticl::assignPCAtoTracksters does. The threads of the block share the layer clusters of the trackster and sum over
    // them in shared memory; after each sum, every thread computes what depends on it.
    struct KernelTracksterProperties {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    HGCalTrackstersSoAAlgoWrapper::Parameters parameters,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    ticl::TracksterDeviceCollection::View tracksters) const {
        constexpr auto block = alpaka::hierarchy::Threads{};
        auto& s = alpaka::declareSharedVar<TracksterSums, __COUNTER__>(acc);
        auto vertices = tracksters.vertices();

        for (auto t : independent_groups(acc, tracksters.energy().metadata().size())) {
          const uint32_t first = vertices.offsets()[t].keys_offsets();
          const uint32_t last = vertices.offsets()[t + 1].keys_offsets();

          if (once_per_block(acc)) {
            s = TracksterSums{};
          }
          alpaka::syncBlockThreads(acc);

          // the energy, the electromagnetic energy, the weighted position and the most energetic layer cluster
          for (auto vertex : independent_group_elements(acc, first, last)) {
            const uint32_t i = vertices.content()[vertex].values();
            const auto position = layerClusters.position()[i];
            const float energy = layerClusters.energy()[i].energy();
            alpaka::atomicAdd(acc, &s.energy, static_cast<double>(energy), block);
            const bool isEm = parameters.barrel ? layerClusters.indexes()[i].seedID().det() == DetId::Ecal
                                                : std::abs(position.z()) <= parameters.zLimitEm;
            if (isEm) {
              alpaka::atomicAdd(acc, &s.emEnergy, static_cast<double>(energy), block);
            }
            alpaka::atomicAdd(acc, &s.position[0], static_cast<double>(energy) * position.x(), block);
            alpaka::atomicAdd(acc, &s.position[1], static_cast<double>(energy) * position.y(), block);
            alpaka::atomicAdd(acc, &s.position[2], static_cast<double>(energy) * position.z(), block);
            alpaka::atomicMax(acc, &s.maxEnergyKey, energyKey(energy, i), block);
          }
          alpaka::syncBlockThreads(acc);

          const float invEnergy = 1.f / static_cast<float>(s.energy);
          const float barycenter[3] = {static_cast<float>(s.position[0]) * invEnergy,
                                       static_cast<float>(s.position[1]) * invEnergy,
                                       static_cast<float>(s.position[2]) * invEnergy};
          const int32_t maxEnergyLayer = layerClusters.position()[layerClusterOfEnergyKey(s.maxEnergyKey)].layer();
          // the slot of a layer in the cleaning window, or -1 outside of it
          auto cleaningSlot = [&](int32_t layer) {
            const int32_t slot = layer - maxEnergyLayer + kCleaningLayers;
            return (slot >= 0 and slot < kCleaningWindow) ? slot : -1;
          };

          // the most energetic layer cluster of each layer of the cleaning window, and the time: the weighted mean of
          // the times of the layer clusters near the axis, corrected for the path to the barycenter
          for (auto vertex : independent_group_elements(acc, first, last)) {
            const uint32_t i = vertices.content()[vertex].values();
            const auto position = layerClusters.position()[i];
            const int32_t slot = parameters.pcaCleaning ? cleaningSlot(position.layer()) : -1;
            if (slot >= 0) {
              alpaka::atomicMax(
                  acc, &s.layerMaxEnergyKey[slot], energyKey(layerClusters.energy()[i].energy(), i), block);
            }

            const auto timing = layerClusters.timing()[i];
            const float point[3] = {position.x(), position.y(), position.z()};
            if (timing.timeError() > 0.f and distance2FromLine(point, barycenter) < kTimeMaxDistance2) {
              const float scale = barycenter[2] * (1.f / position.z()) - 1.f;
              const float deltaZ = barycenter[2] - position.z();
              const float deltaT = 1.f / kSpeedOfLight *
                                   std::sqrt((scale * position.x()) * (scale * position.x()) +
                                             (scale * position.y()) * (scale * position.y()) + deltaZ * deltaZ);
              const float time =
                  std::abs(barycenter[2]) < std::abs(position.z()) ? timing.time() - deltaT : timing.time() + deltaT;
              // in double precision, as in the legacy code (std::pow)
              const float weight =
                  static_cast<float>(1. / (static_cast<double>(timing.timeError()) * timing.timeError()));
              alpaka::atomicAdd(acc, &s.time, static_cast<double>(time) * weight, block);
              alpaka::atomicAdd(acc, &s.timeWeight, static_cast<double>(weight), block);
            }
          }
          alpaka::syncBlockThreads(acc);

          // the PCA uses the most energetic layer cluster of each layer of the cleaning window, or all of them without
          // cleaning
          auto isCleaned = [&](uint32_t i) {
            if (not parameters.pcaCleaning) {
              return true;
            }
            const int32_t slot = cleaningSlot(layerClusters.position()[i].layer());
            return slot >= 0 and layerClusterOfEnergyKey(s.layerMaxEnergyKey[slot]) == i;
          };

          // the PCA needs at least 3 layer clusters; the condition is the same for all the threads of the block
          const bool hasPCA = last - first > 2;
          if (hasPCA) {
            if (parameters.pcaCleaning) {
              for (auto vertex : independent_group_elements(acc, first, last)) {
                const uint32_t i = vertices.content()[vertex].values();
                if (isCleaned(i)) {
                  const auto position = layerClusters.position()[i];
                  const float energy = layerClusters.energy()[i].energy();
                  alpaka::atomicAdd(acc, &s.cleanedEnergy, static_cast<double>(energy), block);
                  alpaka::atomicAdd(acc, &s.cleanedPosition[0], static_cast<double>(energy) * position.x(), block);
                  alpaka::atomicAdd(acc, &s.cleanedPosition[1], static_cast<double>(energy) * position.y(), block);
                  alpaka::atomicAdd(acc, &s.cleanedPosition[2], static_cast<double>(energy) * position.z(), block);
                }
              }
            } else if (once_per_block(acc)) {
              // all the layer clusters are used: the sums are those of the trackster
              s.cleanedEnergy = s.energy;
              for (int k = 0; k < 3; ++k) {
                s.cleanedPosition[k] = s.position[k];
              }
            }
            alpaka::syncBlockThreads(acc);

            // in double precision, as in the legacy code: the PCA then sees the same values on every backend
            const double invCleanedEnergy = 1. / s.cleanedEnergy;
            const Eigen::Vector3d cleanedBarycenter =
                Eigen::Map<const Eigen::Vector3d>(s.cleanedPosition) * invCleanedEnergy;
            auto deviation = [&](uint32_t i) -> Eigen::Vector3d {
              const auto position = layerClusters.position()[i];
              return Eigen::Vector3d(position.x(), position.y(), position.z()) - cleanedBarycenter;
            };

            // the covariance matrix
            for (auto vertex : independent_group_elements(acc, first, last)) {
              const uint32_t i = vertices.content()[vertex].values();
              if (isCleaned(i)) {
                const double weight = layerClusters.energy()[i].energy() * invCleanedEnergy;
                const Eigen::Vector3d d = deviation(i);
                alpaka::atomicAdd(acc, &s.weight2, weight * weight, block);
                for (int a = 0, k = 0; a < 3; ++a) {
                  for (int b = 0; b <= a; ++b, ++k) {
                    alpaka::atomicAdd(acc, &s.covariance[k], weight * d(a) * d(b), block);
                  }
                }
              }
            }
            alpaka::syncBlockThreads(acc);

            if (once_per_block(acc)) {
              Eigen::Matrix3d covariance;
              covariance << s.covariance[0], s.covariance[1], s.covariance[3], s.covariance[1], s.covariance[2],
                  s.covariance[4], s.covariance[3], s.covariance[4], s.covariance[5];
              covariance *= 1. / (1. - s.weight2);
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
              if (solver.info() == Eigen::Success) {
                Eigen::Map<Eigen::Vector3d>(s.eigenvalues) = solver.eigenvalues();
                Eigen::Map<Eigen::Matrix3d>(s.eigenvectors) = solver.eigenvectors();
              }
            }
            alpaka::syncBlockThreads(acc);

            // The spreads along the principal axes, with eigenvectors^T. They equal the square roots of the
            // eigenvalues, but the eigenvalues much smaller than the largest one carry its rounding error, and this
            // sum does not. ticl::assignPCAtoTracksters (CMSSW_20_1_0_pre4) projects with eigenvectors instead.
            const Eigen::Map<const Eigen::Matrix3d> eigenvectors(s.eigenvectors);
            for (auto vertex : independent_group_elements(acc, first, last)) {
              const uint32_t i = vertices.content()[vertex].values();
              if (isCleaned(i)) {
                const double weight = layerClusters.energy()[i].energy() * invCleanedEnergy;
                const Eigen::Vector3d projected = eigenvectors.transpose() * deviation(i);
                for (int k = 0; k < 3; ++k) {
                  alpaka::atomicAdd(acc, &s.sigmasPCA2[k], weight * projected(k) * projected(k), block);
                }
              }
            }
            alpaka::syncBlockThreads(acc);
          }

          if (once_per_block(acc)) {
            auto energy = tracksters.energy()[t];
            energy.rawEnergy() = static_cast<float>(s.energy);
            energy.rawEmEnergy() = static_cast<float>(s.emEnergy);
            energy.regressedEnergy() = 0.f;

            auto position = tracksters.barycenter()[t];
            position.x() = barycenter[0];
            position.y() = barycenter[1];
            position.z() = barycenter[2];

            auto timing = tracksters.timing()[t];
            timing.time() = s.timeWeight > 0. ? static_cast<float>(s.time) / static_cast<float>(s.timeWeight) : kNoTime;
            timing.timeError() =
                s.timeWeight > 0. ? 1.f / std::sqrt(static_cast<float>(s.timeWeight)) : kNoTimeError;

            auto pca = tracksters.pca()[t];
            pca.eigenvalues().setZero();
            pca.eigenvectors().setZero();
            pca.sigmas().setZero();
            pca.sigmasPCA().setZero();
            if (hasPCA) {
              const Eigen::Map<const Eigen::Matrix3d> eigenvectors(s.eigenvectors);
              const double normalisation = 1. - s.weight2;
              // The spreads along x, y, z are the diagonal of the covariance matrix: each layer cluster weighted by its
              // own weight, where ticl::assignPCAtoTracksters (CMSSW_20_1_0_pre4) uses the previous layer cluster's.
              const double diagonal[3] = {s.covariance[0], s.covariance[2], s.covariance[5]};
              // in decreasing eigenvalue order
              for (int k = 0; k < 3; ++k) {
                pca.eigenvalues()(k) = static_cast<float>(s.eigenvalues[2 - k]);
                pca.eigenvectors().col(k) = eigenvectors.col(2 - k).cast<float>();
                pca.sigmas()(k) = static_cast<float>(std::sqrt(diagonal[k] / normalisation));
                pca.sigmasPCA()(k) = static_cast<float>(std::sqrt(s.sigmasPCA2[2 - k] / normalisation));
              }
              // the principal axis points away from the interaction point
              if (pca.eigenvectors()(2, 0) * barycenter[2] < 0.f) {
                pca.eigenvectors().col(0) *= -1.f;
              }
            }

            tracksters.id()[t].idProbabilities().setZero();
          }
          // the shared sums are reset for the next trackster of the block
          alpaka::syncBlockThreads(acc);
        }
      }
    };

  }  // namespace

  void HGCalTrackstersSoAAlgoWrapper::run(Queue& queue,
                                          const Parameters& parameters,
                                          reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                          HGCalSoALayerClustersExtraDeviceCollection::ConstView layerClustersClustering,
                                          ticl::TracksterDeviceCollection::View tracksters) const {
    // the layer clusters of the first trackster start at 0, also when there is no trackster
    uint32_t* offsets = tracksters.vertices().offsets().keys_offsets().data();
    auto firstOffset = make_device_view(queue, offsets, 1);
    alpaka::memset(queue, firstOffset, 0x0);
    const uint32_t nTracksters = tracksters.energy().metadata().size();
    if (nTracksters == 0) {
      return;
    }
    const uint32_t nLayerClusters = layerClustersClustering.metadata().size();

    auto tracksterSizes = make_device_buffer<uint32_t[]>(queue, nTracksters);
    alpaka::memset(queue, tracksterSizes, 0x0);

    constexpr uint32_t blockSize = 256;
    const auto workDivLayerClusters = make_workdiv<Acc1D>(divide_up_by(nLayerClusters, blockSize), blockSize);
    alpaka::exec<Acc1D>(
        queue, workDivLayerClusters, KernelTracksterSizes{}, layerClustersClustering, tracksterSizes.data());
    // inclusive scan of the sizes, so offsets[t + 1] is where the layer clusters of trackster t end. The scan leaves
    // the sizes unchanged.
    iterativePrefixScan<Acc1D>(tracksterSizes.data(), offsets + 1, nTracksters, queue);
    // single thread: the layer clusters of a trackster are then written in the same order on every backend (the
    // atomic countdown would order them by thread scheduling)
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(1, 1),
                        KernelTracksterVertices{},
                        layerClustersClustering,
                        tracksterSizes.data(),
                        tracksters);

    // One thread per trackster: the sums over its layer clusters then accumulate in their order on every backend
    // (the block atomics would order them by thread scheduling). Most tracksters have a few layer clusters and the
    // largest a few hundred, and the tracksters run in parallel.
    const auto workDivTracksters = make_workdiv<Acc1D>(nTracksters, 1);
    alpaka::exec<Acc1D>(queue, workDivTracksters, KernelTracksterProperties{}, parameters, layerClusters, tracksters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
