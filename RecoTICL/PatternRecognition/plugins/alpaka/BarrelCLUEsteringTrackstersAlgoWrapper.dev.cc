#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>

#include <alpaka/alpaka.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "BarrelCLUEsteringTrackstersAlgoWrapper.h"

#include "CLUEstering/core/detail/defines.hpp"
#include "CLUEstering/core/Clusterer.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  // The points are the layer clusters, in their order, followed by kGhosts ghost points. Per point: the inputs of
  // CLUEstering and its output (the cluster index). The coordinates are eta / sigmaRef and phi / sigmaRef, with phi in
  // [0, 2 pi) and sigmaRef the smallest sigmaT; sigmaT is the one of the barrel calorimeter of the point, relative to
  // sigmaRef. The layer enters only the distance: the barrel has a few layers, which the tiles would not separate.
  // Outside of the anonymous namespace, as nvcc warns about the unused members of a SoA layout with internal linkage.
  GENERATE_SOA_LAYOUT(BarrelCLUEsteringTrackstersPointsLayout,
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, weight),
                      SOA_COLUMN(float, rhocScale),
                      SOA_COLUMN(uint32_t, tag),
                      SOA_COLUMN(int32_t, layer),
                      SOA_COLUMN(float, sigmaT),
                      SOA_COLUMN(float, searchSigma),
                      SOA_COLUMN(int, clusterIndex))

  namespace {

    constexpr uint32_t kComponents = BarrelCLUEsteringTrackstersAlgoWrapper::kComponents;
    constexpr int kInvalidCluster = -1;
    constexpr int kInvalidTrackster = -1;
    constexpr float kTwoPi = 2.f * std::numbers::pi_v<float>;

    // The layer of the points that are no neighbour of any other: the ghosts, and the layer clusters without a
    // position or energy
    constexpr int32_t kIsolated = -1;

    // CLUEstering wraps phi with the range of the coordinates, not with the period, and the search windows wider than
    // half that range miss the tiles in their middle. Two ghost points at phi = 0 and phi = 2 pi make the range the
    // period. They weigh nothing and are isolated, so they belong to no trackster.
    constexpr uint32_t kGhosts = 2;

    using PointsDeviceCollection = PortableCollection<BarrelCLUEsteringTrackstersPointsLayout<>>;

    // The barrel calorimeter of a layer cluster, from the DetId of its seed: the barrel layer clusters are either
    // ECAL or HCAL ones
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t component(const DetId seed) { return seed.det() == DetId::Hcal ? 1 : 0; }

    // The clustering distance: max(transverse distance / mean sigmaT of the pair, layer distance / layerScale), with
    // phi periodic. Its unit ball is a disc times an interval along the layers, and the pairs of the coarser barrel
    // calorimeter see their transverse distance shrunk. The isolated points, and the different points of a layer
    // unless sameLayerNeighbours, are out of reach.
    struct BarrelMetric {
      using value_type = float;

      const float* sigmaT;
      const int32_t* layer;
      float phiPeriod;
      float invLayerScale;
      bool sameLayerNeighbours;

      template <typename TPoints>
      ALPAKA_FN_HOST_ACC float operator()(const TPoints& points, std::size_t i, std::size_t j) const {
        if (i == j) {
          return 0.f;
        }
        const int32_t layerI = layer[i];
        const int32_t layerJ = layer[j];
        if (layerI == kIsolated or layerJ == kIsolated or (layerI == layerJ and not sameLayerNeighbours)) {
          return std::numeric_limits<float>::infinity();
        }
        const auto pi = points[static_cast<int>(i)];
        const auto pj = points[static_cast<int>(j)];
        // in double precision and rounded to float once, like the coordinates: CLUEstering then compares the same
        // values on every backend, and its distance ties break deterministically
        const double dEta = pi[0] - pj[0];
        const double dPhi = std::abs(pi[1] - pj[1]);
        const double dPhiWrapped = std::min(dPhi, static_cast<double>(phiPeriod) - dPhi);
        const double transverse = std::sqrt(dEta * dEta + dPhiWrapped * dPhiWrapped) * 2.0 / (sigmaT[i] + sigmaT[j]);
        return static_cast<float>(std::max(transverse, std::abs(double(layerI) - double(layerJ)) * invLayerScale));
      }
    };

    // Per layer cluster -> per point: the clustering coordinates, weight, seed threshold scale, tag (the seed DetId,
    // to break the ties deterministically), layer, sigmaT and the sigma of the search window of the point; then the
    // ghosts.
    struct KernelFillPoints {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    float invSigmaRef,
                                    std::array<float, kComponents> rhocScale,
                                    std::array<float, kComponents> relativeSigmaT,
                                    std::array<float, kComponents> searchSigma,
                                    reco::CaloClusterDeviceCollection::ConstView layerClusters,
                                    PointsDeviceCollection::View points) const {
        const uint32_t nLayerClusters = layerClusters.position().metadata().size();
        for (auto i : uniform_elements(acc, nLayerClusters)) {
          const auto position = layerClusters.position()[i];
          const float energy = layerClusters.energy()[i].energy();
          const DetId seed = layerClusters.indexes()[i].seedID();
          const uint32_t c = component(seed);
          // The coordinates are computed in double precision and rounded to float once: the float store removes
          // the 1-ULP differences between the transcendentals of the backends, which would otherwise flip the
          // clustering decisions of the tie-rich barrel layer clusters
          const double px = position.x();
          const double py = position.y();
          const double r = std::sqrt(px * px + py * py);
          // a layer cluster without energy is at the origin, where eta is not defined
          const bool isolated = not(energy > 0.f and r > 0.);
          double phi = std::atan2(py, px);
          if (phi < 0.) {
            phi += kTwoPi;
          }
          points[i].eta() = isolated ? 0.f : static_cast<float>(std::asinh(position.z() / r) * invSigmaRef);
          points[i].phi() = isolated ? 0.f : static_cast<float>(phi * invSigmaRef);
          points[i].weight() = isolated ? 0.f : energy;
          points[i].rhocScale() = rhocScale[c];
          points[i].tag() = seed.rawId();
          points[i].layer() = isolated ? kIsolated : position.layer();
          points[i].sigmaT() = relativeSigmaT[c];
          points[i].searchSigma() = searchSigma[c];
        }

        if (once_per_grid(acc)) {
          for (uint32_t g = 0; g < kGhosts; ++g) {
            auto ghost = points[nLayerClusters + g];
            ghost.eta() = 0.f;
            ghost.phi() = g * kTwoPi * invSigmaRef;
            ghost.weight() = 0.f;
            ghost.rhocScale() = 1.f;
            ghost.tag() = g;
            ghost.layer() = kIsolated;
            ghost.sigmaT() = 1.f;
            ghost.searchSigma() = 0.f;
          }
        }
      }
    };

    // Per layer cluster -> per cluster: the number of layer clusters. The isolated ones count for none, so that a
    // cluster of them never is a trackster.
    struct KernelCountClusterSizes {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t nLayerClusters,
                                    PointsDeviceCollection::ConstView points,
                                    uint32_t* clusterSizes) const {
        for (auto i : uniform_elements(acc, nLayerClusters)) {
          const int cluster = points[i].clusterIndex();
          if (cluster != kInvalidCluster and points[i].layer() != kIsolated) {
            alpaka::atomicAdd(acc, &clusterSizes[cluster], 1u);
          }
        }
      }
    };

    // Per cluster: whether it is a trackster, having at least minLayerClusters (>= 1) layer clusters, and the number
    // of layer clusters in the tracksters
    struct KernelSelectTracksters {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t nClusters,
                                    uint32_t minLayerClusters,
                                    const uint32_t* clusterSizes,
                                    uint32_t* isTrackster,
                                    HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const {
        for (auto c : uniform_elements(acc, nClusters)) {
          const bool selected = clusterSizes[c] >= minLayerClusters;
          isTrackster[c] = selected;
          if (selected) {
            alpaka::atomicAdd(acc, &layerClustersClustering.numberOfVerticesScalar(), clusterSizes[c]);
          }
        }
      }
    };

    // Per layer cluster: its trackster, if its cluster is a trackster. tracksterEnd[c] - 1 is the trackster of cluster
    // c, so the tracksters keep the order of the clusters.
    struct KernelWriteLayerClustersClustering {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    uint32_t nPoints,
                                    PointsDeviceCollection::ConstView points,
                                    const uint32_t* isTrackster,
                                    const uint32_t* tracksterEnd,
                                    HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const {
        if (once_per_grid(acc)) {
          layerClustersClustering.numberOfTrackstersScalar() = tracksterEnd[nPoints - 1];
        }
        for (auto i : uniform_elements(acc, layerClustersClustering.metadata().size())) {
          const int cluster = points[i].clusterIndex();
          layerClustersClustering[i].tracksterIndex() =
              (cluster != kInvalidCluster and isTrackster[cluster]) ? static_cast<int>(tracksterEnd[cluster] - 1)
                                                                    : kInvalidTrackster;
        }
      }
    };

  }  // namespace

  void BarrelCLUEsteringTrackstersAlgoWrapper::run(
      Queue& queue,
      const Parameters& parameters,
      reco::CaloClusterDeviceCollection::ConstView layerClusters,
      HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const {
    auto nTracksters = make_device_view<unsigned int>(queue, layerClustersClustering.numberOfTrackstersScalar());
    alpaka::memset(queue, nTracksters, 0x0);
    auto nVertices = make_device_view<unsigned int>(queue, layerClustersClustering.numberOfVerticesScalar());
    alpaka::memset(queue, nVertices, 0x0);
    const uint32_t nLayerClusters = layerClusters.position().metadata().size();
    if (nLayerClusters == 0) {
      return;
    }

    // The coordinates are in units of the smallest sigmaT, and the sigmaT of the barrel calorimeters relative to it
    const float sigmaRef = std::ranges::min(parameters.sigmaT);
    const float maxRelativeSigmaT = std::ranges::max(parameters.sigmaT) / sigmaRef;
    std::array<float, kComponents> relativeSigmaT;
    std::array<float, kComponents> searchSigma;
    for (uint32_t c = 0; c < kComponents; ++c) {
      relativeSigmaT[c] = parameters.sigmaT[c] / sigmaRef;
      // A neighbour j is within the distance R of point i only if each transverse coordinate differs by at most
      // R * (sigmaT_i + sigmaT_j) / 2, and CLUEstering searches within R * searchSigma * sqrt(2). Every layer is within
      // reach of the others, so the window covers the coarsest barrel calorimeter. 1 / sqrt(2) rounded up, so that the
      // window is never narrower than needed.
      constexpr float invSqrt2 = 0.70710681f;
      searchSigma[c] = 0.5f * (relativeSigmaT[c] + maxRelativeSigmaT) * invSqrt2;
    }

    // CLUEstering seeds a point above min_density times its density uncertainty, and searches its nearest higher
    // within seedingDistance above min_density, within outlierDistance below. The seed threshold of a layer cluster is
    // the rhoc of its barrel calorimeter: min_density is the rhoc of EB and the density uncertainty of a point is its
    // threshold relative to it, so the nearest higher search switches at the seed threshold when EB and HB share their
    // rhoc, as in the legacy barrel CLUE3D
    const float rhocPivot = parameters.rhoc[0];
    std::array<float, kComponents> rhocScale;
    for (uint32_t c = 0; c < kComponents; ++c) {
      rhocScale[c] = parameters.rhoc[c] / rhocPivot;
    }

    // The points: the layer clusters and the ghosts. Their number is known on the host.
    const uint32_t nPoints = nLayerClusters + kGhosts;
    PointsDeviceCollection points(queue, nPoints);

    constexpr uint32_t blockSize = 256;
    const auto workDivLayerClusters = make_workdiv<Acc1D>(divide_up_by(nLayerClusters, blockSize), blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelFillPoints{},
                        1.f / sigmaRef,
                        rhocScale,
                        relativeSigmaT,
                        searchSigma,
                        layerClusters,
                        points.view());
    // CLUEstering reduces the coordinates to the tiles' extents outside of the queue: the points must be filled
    alpaka::wait(queue);

    auto pointsView = points.view();
    clue::PointsDevice<2, float> clusteringPoints(queue,
                                                  static_cast<int32_t>(nPoints),
                                                  pointsView.eta().data(),
                                                  pointsView.phi().data(),
                                                  pointsView.weight().data(),
                                                  pointsView.clusterIndex().data());
    clusteringPoints.set_tags(std::span<const uint32_t>(pointsView.tag().data(), nPoints));
    // the seed threshold of a point is min_density times its density uncertainty: the rhoc of its barrel calorimeter
    clusteringPoints.set_density_uncertainty(std::span<const float>(pointsView.rhocScale().data(), nPoints));
    // the per point sigma of the transverse coordinates sets the search window
    clusteringPoints.set_sigma(0, std::span<const float>(pointsView.searchSigma().data(), nPoints));
    clusteringPoints.set_sigma(1, std::span<const float>(pointsView.searchSigma().data(), nPoints));

    clue::Clusterer<2> algo(
        queue, parameters.deltac, rhocPivot, parameters.outlierDistance, parameters.seedingDistance);
    algo.setWrappedCoordinates(0, 1);
    algo.make_clusters(queue,
                       clusteringPoints,
                       BarrelMetric{.sigmaT = pointsView.sigmaT().data(),
                                    .layer = pointsView.layer().data(),
                                    .phiPeriod = kTwoPi / sigmaRef,
                                    .invLayerScale = 1.f / parameters.layerScale,
                                    .sameLayerNeighbours = parameters.sameLayerNeighbours});

    // There are at most as many clusters as points: the sizes of the unused slots stay 0, below any minLayerClusters
    auto clusterSizes = make_device_buffer<uint32_t[]>(queue, nPoints);
    alpaka::memset(queue, clusterSizes, 0x0);
    auto isTrackster = make_device_buffer<uint32_t[]>(queue, nPoints);
    auto tracksterEnd = make_device_buffer<uint32_t[]>(queue, nPoints);

    const auto workDivPoints = make_workdiv<Acc1D>(divide_up_by(nPoints, blockSize), blockSize);
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelCountClusterSizes{},
                        nLayerClusters,
                        points.const_view(),
                        clusterSizes.data());
    alpaka::exec<Acc1D>(queue,
                        workDivPoints,
                        KernelSelectTracksters{},
                        nPoints,
                        std::max(parameters.minLayerClusters, 1u),
                        clusterSizes.data(),
                        isTrackster.data(),
                        layerClustersClustering);
    // inclusive scan of the trackster flags
    iterativePrefixScan<Acc1D>(isTrackster.data(), tracksterEnd.data(), nPoints, queue);
    alpaka::exec<Acc1D>(queue,
                        workDivLayerClusters,
                        KernelWriteLayerClustersClustering{},
                        nPoints,
                        points.const_view(),
                        isTrackster.data(),
                        tracksterEnd.data(),
                        layerClustersClustering);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
