#ifndef RecoParticleFlow_PFRecHitProducer_plugins_alpaka_PFRecHitProducerKernel_h
#define RecoParticleFlow_PFRecHitProducer_plugins_alpaka_PFRecHitProducerKernel_h

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "CalorimeterDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename CAL>
  class PFRecHitProducerKernel {
  public:
    PFRecHitProducerKernel(Queue& queue, const uint32_t num_recHits);

    // Run kernel: apply filters to rec hits and construct PF rec hits
    // This may be executed multiple times per event
    void processRecHits(Queue& queue,
                        const typename CAL::CaloRecHitSoATypeDevice& recHits,
                        const typename CAL::ParameterType& params,
                        const typename CAL::TopologyTypeDevice& topology,
                        reco::PFRecHitDeviceCollection& pfRecHits);

    // Run kernel: Associate topology information (position, neighbours)
    void associateTopologyInfo(Queue& queue,
                               const typename CAL::TopologyTypeDevice& topology,
                               reco::PFRecHitDeviceCollection& pfRecHits);

  private:
    cms::alpakatools::device_buffer<Device, uint32_t[]> denseId2pfRecHit_;
    cms::alpakatools::device_buffer<Device, uint32_t> num_pfRecHits_;
    // per input recHit: the acceptance flag and its inclusive slot scan, so that the PFRecHits are packed in the
    // order of the input recHits on every backend (an atomic counter would order them by thread scheduling);
    // base is where the current input collection starts in the shared output
    cms::alpakatools::device_buffer<Device, uint32_t[]> accept_;
    cms::alpakatools::device_buffer<Device, uint32_t[]> slotEnd_;
    cms::alpakatools::device_buffer<Device, uint32_t> base_;
    WorkDiv<Dim1D> work_div_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoParticleFlow_PFRecHitProducer_plugins_alpaka_PFRecHitProducerKernel_h
