#include <iostream>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEM.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUEGPUProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUECPUProduct.h"

class HGCalLayerClusterProducerEMGPUtoSoA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HGCalLayerClusterProducerEMGPUtoSoA(const edm::ParameterSet& ps);
  ~HGCalLayerClusterProducerEMGPUtoSoA() override;

  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  cms::cuda::ContextState ctxState_;
  edm::EDGetTokenT<cms::cuda::Product<HGCCLUEGPUProduct>> clueGPUToken_;
  edm::EDPutTokenT<HGCCLUECPUProduct> clueCPUSoAToken_;

  std::unique_ptr<HGCCLUECPUProduct> prodPtr_;
  std::unique_ptr<HGCalCLUEAlgoGPUEM> mAlgo;
};

HGCalLayerClusterProducerEMGPUtoSoA::HGCalLayerClusterProducerEMGPUtoSoA(const edm::ParameterSet& ps)
  : clueGPUToken_{consumes<cms::cuda::Product<HGCCLUEGPUProduct>>(
          ps.getParameter<edm::InputTag>("EMInputCLUEGPU"))},
      clueCPUSoAToken_(produces<HGCCLUECPUProduct>()) {}

HGCalLayerClusterProducerEMGPUtoSoA::~HGCalLayerClusterProducerEMGPUtoSoA() {}

void HGCalLayerClusterProducerEMGPUtoSoA::acquire(edm::Event const& event,
                               edm::EventSetup const& setup,
                               edm::WaitingTaskWithArenaHolder w) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w)};
  const auto& gpuCLUEHits = ctx.get(event, clueGPUToken_);
  const unsigned nhits(gpuCLUEHits.nHits());

  prodPtr_ = std::make_unique<HGCCLUECPUProduct>(nhits, ctx.stream());

  HGCCLUECPUProduct& prod_ = *prodPtr_;

  mAlgo = std::make_unique<HGCalCLUEAlgoGPUEM>(prod_.get(), gpuCLUEHits.get(), nhits);
  mAlgo->copy_tohost(ctx.stream());
}

void HGCalLayerClusterProducerEMGPUtoSoA::produce(edm::Event& event, const edm::EventSetup& setup) { event.put(std::move(prodPtr_)); }

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterProducerEMGPUtoSoA);
