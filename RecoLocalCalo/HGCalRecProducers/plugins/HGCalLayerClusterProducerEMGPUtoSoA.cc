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

#include "CUDADataFormats/HGCal/interface/HGCCLUEGPUHitsProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUECPUHitsProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUEGPUClustersProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUECPUClustersProduct.h"

class HGCalLayerClusterProducerEMGPUtoSoA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit HGCalLayerClusterProducerEMGPUtoSoA(const edm::ParameterSet& ps);
  ~HGCalLayerClusterProducerEMGPUtoSoA() override;

  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  cms::cuda::ContextState ctxState_;
  edm::EDGetTokenT<cms::cuda::Product<HGCCLUEGPUHitsProduct>> clueGPUHitsToken_;
  edm::EDGetTokenT<cms::cuda::Product<HGCCLUEGPUClustersProduct>> clueGPUClustersToken_;
  edm::EDPutTokenT<HGCCLUECPUHitsProduct> clueCPUHitsSoAToken_;
  edm::EDPutTokenT<HGCCLUECPUClustersProduct> clueCPUClustersSoAToken_;

  std::unique_ptr<HGCCLUECPUHitsProduct> prodHitsPtr_;
  std::unique_ptr<HGCCLUECPUClustersProduct> prodClustersPtr_;
  std::unique_ptr<HGCalCLUEAlgoGPUEM> mAlgo;
};

HGCalLayerClusterProducerEMGPUtoSoA::HGCalLayerClusterProducerEMGPUtoSoA(const edm::ParameterSet& ps)
  : clueGPUHitsToken_{consumes<cms::cuda::Product<HGCCLUEGPUHitsProduct>>(
          ps.getParameter<edm::InputTag>("EMInputCLUEHitsGPU"))},
    clueGPUClustersToken_{consumes<cms::cuda::Product<HGCCLUEGPUClustersProduct>>(
          ps.getParameter<edm::InputTag>("EMInputCLUEClustersGPU"))},
    clueCPUHitsSoAToken_(produces<HGCCLUECPUHitsProduct>()),
    clueCPUClustersSoAToken_(produces<HGCCLUECPUClustersProduct>())
{}

HGCalLayerClusterProducerEMGPUtoSoA::~HGCalLayerClusterProducerEMGPUtoSoA() {}

void HGCalLayerClusterProducerEMGPUtoSoA::acquire(edm::Event const& event,
                               edm::EventSetup const& setup,
                               edm::WaitingTaskWithArenaHolder w) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w)};
  const auto& gpuCLUEHits = ctx.get(event, clueGPUHitsToken_);
  const auto& gpuCLUEClusters = ctx.get(event, clueGPUClustersToken_);
  const unsigned nhits(gpuCLUEHits.nHits());
  const unsigned nclusters(gpuCLUEClusters.nClusters());
  
  prodHitsPtr_ = std::make_unique<HGCCLUECPUHitsProduct>(nhits, ctx.stream());
  prodClustersPtr_ = std::make_unique<HGCCLUECPUHitsProduct>(nhits, ctx.stream());

  HGCCLUECPUHitsProduct& prodHits_ = *prodHitsPtr_;
  HGCCLUECPUHitsProduct& prodClusters_ = *prodClustersPtr_;

  mAlgo = std::make_unique<HGCalCLUEAlgoGPUEM>(prodHits_.get(), gpuCLUEHits.get(),
					       prodClusters_.get(), gpuCLUEClusters.get(),
					       nhits, nclusters);
  mAlgo->copy_tohost(ctx.stream());
}

void HGCalLayerClusterProducerEMGPUtoSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  event.put(std::move(prodHitsPtr_));
  event.put(std::move(prodClustersPtr_));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterProducerEMGPUtoSoA);
