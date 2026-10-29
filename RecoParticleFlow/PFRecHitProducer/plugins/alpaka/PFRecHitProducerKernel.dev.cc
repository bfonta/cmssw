#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "PFRecHitProducerKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace particleFlowRecHitProducer;

  // Kernel to apply cuts to calorimeter hits and construct PFRecHits
  template <typename CAL>
  struct PFRecHitProducerKernelConstruct {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const typename CAL::ParameterType::ConstView params,
                                  const typename CAL::TopologyTypeDevice::ConstView topology,
                                  const typename CAL::CaloRecHitSoATypeDevice::ConstView recHits,
                                  reco::PFRecHitDeviceCollection::View pfRecHits,
                                  uint32_t* __restrict__ denseId2pfRecHit,
                                  const uint32_t* __restrict__ accept,
                                  const uint32_t* __restrict__ slotEnd,
                                  const uint32_t* __restrict__ base) const {
      // Strided loop over CaloRecHits
      for (int32_t i : cms::alpakatools::uniform_elements(acc, recHits.metadata().size())) {
        if (accept[i] == 0)
          continue;

        // The inclusive scan of the acceptance flags gives a unique consecutive index to every kept recHit, in the
        // order of the input recHits on every backend (an atomic counter would order them by thread scheduling),
        // after the PFRecHits of the preceding input collections
        const uint32_t j = *base + slotEnd[i] - 1;

        // Construct PFRecHit from CAL recHit (specialised for HCAL/ECAL)
        constructPFRecHit(pfRecHits[j], recHits[i]);

        // Fill denseId -> pfRecHit index map
        denseId2pfRecHit[CAL::detId2denseId(pfRecHits.detId(j))] = j;
      }
    }

    ALPAKA_FN_ACC static bool applyCuts(const typename CAL::CaloRecHitSoATypeDevice::ConstView::const_element rh,
                                        const typename CAL::ParameterType::ConstView params,
                                        const typename CAL::TopologyTypeDevice::ConstView topology);

    ALPAKA_FN_ACC static void constructPFRecHit(
        reco::PFRecHitDeviceCollection::View::element pfrh,
        const typename CAL::CaloRecHitSoATypeDevice::ConstView::const_element rh);
  };

  // Kernel to record where the PFRecHits of one input collection start in the shared output: the collections are
  // packed one after the other, and the total is the number of PFRecHits for the topology kernel
  struct KernelShiftBase {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const uint32_t* __restrict__ last,
                                  uint32_t* __restrict__ num_pfRecHits,
                                  uint32_t* __restrict__ base) const {
      if (cms::alpakatools::once_per_grid(acc)) {
        *base = *num_pfRecHits;
        *num_pfRecHits += *last;
      }
    }
  };

  // Kernel to apply the quality cuts: accept[i] is 1 if the input recHit i is kept
  template <typename CAL>
  struct PFRecHitProducerKernelAccept {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const typename CAL::ParameterType::ConstView params,
                                  const typename CAL::TopologyTypeDevice::ConstView topology,
                                  const typename CAL::CaloRecHitSoATypeDevice::ConstView recHits,
                                  uint32_t* __restrict__ accept) const {
      for (int32_t i : cms::alpakatools::uniform_elements(acc, recHits.metadata().size())) {
        accept[i] = PFRecHitProducerKernelConstruct<CAL>::applyCuts(recHits[i], params, topology) ? 1 : 0;
      }
    }
  };

  template <>
  ALPAKA_FN_ACC bool PFRecHitProducerKernelConstruct<HCAL>::applyCuts(
      const typename HCAL::CaloRecHitSoATypeDevice::ConstView::const_element rh,
      const HCAL::ParameterType::ConstView params,
      const HCAL::TopologyTypeDevice::ConstView topology) {
    // Reject HCAL recHits below enery threshold
    float threshold = 9999.;
    const uint32_t detId = rh.detId();
    const uint32_t depth = HCAL::getDepth(detId);
    const uint32_t subdet = getSubdet(detId);

    // skip bad channels
    if (rh.chi2() < 0)
      return false;

    if (topology.cutsFromDB()) {
      const auto& denseId = HCAL::detId2denseId(detId);
      if (denseId != HCAL::kInvalidDenseId) {
        threshold = topology.noiseThreshold()[denseId];
      } else {
        printf("Encountered invalid denseId for detId %u (subdetector %u)!", detId, subdet);
        return false;
      }
    } else {
      if (subdet == HcalBarrel) {
        threshold = params.energyThresholds()[depth - 1];
      } else if (subdet == HcalEndcap) {
        threshold = params.energyThresholds()[depth - 1 + HCAL::kMaxDepthHB];
      } else {
        printf("Rechit with detId %u has invalid subdetector %u!\n", detId, subdet);
        return false;
      }
    }
    return rh.energy() >= threshold;
  }

  template <>
  ALPAKA_FN_ACC bool PFRecHitProducerKernelConstruct<ECAL>::applyCuts(
      const ECAL::CaloRecHitSoATypeDevice::ConstView::const_element rh,
      const ECAL::ParameterType::ConstView params,
      const ECAL::TopologyTypeDevice::ConstView topology) {
    // Reject ECAL recHits below energy threshold
    if (rh.energy() < params.energyThresholds()[ECAL::detId2denseId(rh.detId())])
      return false;

    // Reject ECAL recHits of bad quality
    if ((ECAL::checkFlag(rh.flags(), ECAL::Flags::kOutOfTime) && rh.energy() > params.cleaningThreshold()) ||
        ECAL::checkFlag(rh.flags(), ECAL::Flags::kTowerRecovered) || ECAL::checkFlag(rh.flags(), ECAL::Flags::kWeird) ||
        ECAL::checkFlag(rh.flags(), ECAL::Flags::kDiWeird))
      return false;

    return true;
  }

  template <>
  ALPAKA_FN_ACC void PFRecHitProducerKernelConstruct<HCAL>::constructPFRecHit(
      reco::PFRecHitDeviceCollection::View::element pfrh,
      const HCAL::CaloRecHitSoATypeDevice::ConstView::const_element rh) {
    pfrh.detId() = rh.detId();
    pfrh.denseId() = HCAL::detId2denseId(rh.detId());
    pfrh.energy() = rh.energy();
    pfrh.time() = rh.timeM0();
    pfrh.depth() = HCAL::getDepth(pfrh.detId());
    const uint32_t subdet = getSubdet(pfrh.detId());
    if (subdet == HcalBarrel)
      pfrh.layer() = PFLayer::HCAL_BARREL1;
    else if (subdet == HcalEndcap)
      pfrh.layer() = PFLayer::HCAL_ENDCAP;
    else
      pfrh.layer() = PFLayer::NONE;
  }

  template <>
  ALPAKA_FN_ACC void PFRecHitProducerKernelConstruct<ECAL>::constructPFRecHit(
      reco::PFRecHitDeviceCollection::View::element pfrh,
      const ECAL::CaloRecHitSoATypeDevice::ConstView::const_element rh) {
    pfrh.detId() = rh.detId();
    pfrh.denseId() = ECAL::detId2denseId(rh.detId());
    pfrh.energy() = rh.energy();
    pfrh.time() = rh.time();
    pfrh.depth() = 1;
    const uint32_t subdet = getSubdet(pfrh.detId());
    if (subdet == EcalBarrel)
      pfrh.layer() = PFLayer::ECAL_BARREL;
    else if (subdet == EcalEndcap)
      pfrh.layer() = PFLayer::ECAL_ENDCAP;
    else
      pfrh.layer() = PFLayer::NONE;
  }

  // Kernel to associate topology information of PFRecHits
  template <typename CAL>
  struct PFRecHitProducerKernelTopology {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const typename CAL::TopologyTypeDevice::ConstView topology,
                                  reco::PFRecHitDeviceCollection::View pfRecHits,
                                  const uint32_t* __restrict__ denseId2pfRecHit,
                                  uint32_t* __restrict__ num_pfRecHits) const {
      // First thread updates size field pfRecHits SoA
      if (const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]; thread == 0)
        pfRecHits.size() = *num_pfRecHits;

      // Assign position information and associate neighbours
      for (int32_t i : cms::alpakatools::uniform_elements(acc, *num_pfRecHits)) {
        const uint32_t denseId = CAL::detId2denseId(pfRecHits.detId(i));

        pfRecHits.x(i) = topology.positionX(denseId);
        pfRecHits.y(i) = topology.positionY(denseId);
        pfRecHits.z(i) = topology.positionZ(denseId);

        for (uint32_t n = 0; n < 8; n++) {
          pfRecHits.neighbours(i)(n) = -1;
          const uint32_t denseId_neighbour = topology.neighbours(denseId)(n);
          if (denseId_neighbour != CAL::kInvalidDenseId) {
            const uint32_t pfRecHit_neighbour = denseId2pfRecHit[denseId_neighbour];
            if (pfRecHit_neighbour != 0xffffffff)
              pfRecHits.neighbours(i)(n) = (int32_t)pfRecHit_neighbour;
          }
        }
      }
    }
  };

  template <typename CAL>
  PFRecHitProducerKernel<CAL>::PFRecHitProducerKernel(Queue& queue, const uint32_t num_recHits)
      : denseId2pfRecHit_(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, CAL::kSize)),
        num_pfRecHits_(cms::alpakatools::make_device_buffer<uint32_t>(queue)),
        accept_(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, num_recHits)),
        slotEnd_(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, num_recHits)),
        base_(cms::alpakatools::make_device_buffer<uint32_t>(queue)),
        work_div_(cms::alpakatools::make_workdiv<Acc1D>(1, 1)) {
    alpaka::memset(queue, denseId2pfRecHit_, 0xff);  // Reset denseId -> pfRecHit index map
    alpaka::memset(queue, num_pfRecHits_, 0x00);     // Reset global pfRecHit counter

    const uint32_t items = 64;
    const uint32_t groups = cms::alpakatools::divide_up_by(num_recHits, items);
    work_div_ = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
  }

  template <typename CAL>
  void PFRecHitProducerKernel<CAL>::processRecHits(Queue& queue,
                                                   const typename CAL::CaloRecHitSoATypeDevice& recHits,
                                                   const typename CAL::ParameterType& params,
                                                   const typename CAL::TopologyTypeDevice& topology,
                                                   reco::PFRecHitDeviceCollection& pfRecHits) {
    const uint32_t n = recHits.view().metadata().size();
    alpaka::exec<Acc1D>(
        queue, work_div_, PFRecHitProducerKernelAccept<CAL>{}, params.view(), topology.view(), recHits.view(), accept_.data());
    if (n > 0) {
      // inclusive scan of the acceptance flags: kept recHits are packed in the order of the input recHits
      cms::alpakatools::iterativePrefixScan<Acc1D>(accept_.data(), slotEnd_.data(), n, queue);
      // the collections are packed one after the other in the shared output: where this one starts
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                          KernelShiftBase{},
                          slotEnd_.data() + n - 1,
                          num_pfRecHits_.data(),
                          base_.data());
    }
    alpaka::exec<Acc1D>(queue,
                        work_div_,
                        PFRecHitProducerKernelConstruct<CAL>{},
                        params.view(),
                        topology.view(),
                        recHits.view(),
                        pfRecHits.view(),
                        denseId2pfRecHit_.data(),
                        accept_.data(),
                        slotEnd_.data(),
                        base_.data());
  }

  template <typename CAL>
  void PFRecHitProducerKernel<CAL>::associateTopologyInfo(Queue& queue,
                                                          const typename CAL::TopologyTypeDevice& topology,
                                                          reco::PFRecHitDeviceCollection& pfRecHits) {
    alpaka::exec<Acc1D>(queue,
                        work_div_,
                        PFRecHitProducerKernelTopology<CAL>{},
                        topology.view(),
                        pfRecHits.view(),
                        denseId2pfRecHit_.data(),
                        num_pfRecHits_.data());
  }

  // Instantiate templates
  template class PFRecHitProducerKernel<HCAL>;
  template class PFRecHitProducerKernel<ECAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
