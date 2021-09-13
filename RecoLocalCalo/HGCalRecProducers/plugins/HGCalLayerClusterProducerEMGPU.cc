#ifndef __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducerEMGPU_H__
#define __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducerEMGPU_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEM.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUEGPUHitsProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUEGPUClustersProduct.h"

#include "CondFormats/HGCalObjects/interface/HeterogeneousHGCalPositionsConditions.h"
#include "CondFormats/DataRecord/interface/HeterogeneousHGCalPositionsConditionsRecord.h"


class HGCalLayerClusterProducerEMGPU : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  HGCalLayerClusterProducerEMGPU(const edm::ParameterSet&);
  ~HGCalLayerClusterProducerEMGPU() override {}
  // static void fillDescriptions(edm::ConfigurationDescriptions& descriptions); 

  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  float mDc, mKappa, mEcut, mOutlierDeltaFactor;
  edm::ESGetToken<HeterogeneousHGCalPositionsConditions,
		  HeterogeneousHGCalPositionsConditionsRecord> gpuPositionsTok_;
  edm::EDGetTokenT<cms::cuda::Product<HGCRecHitGPUProduct>> InEEToken;
  edm::EDPutTokenT<cms::cuda::Product<HGCCLUEGPUHitsProduct>> OutEEHitsToken;
  edm::EDPutTokenT<cms::cuda::Product<HGCCLUEGPUClustersProduct>> OutEEClustersToken;
  cms::cuda::ContextState ctxState_;
  HGCCLUEGPUHitsProduct mCLUEHits;
  HGCCLUEGPUClustersProduct mCLUEClusters;
  std::unique_ptr<HGCalCLUEAlgoGPUEM> mAlgo;
};

DEFINE_FWK_MODULE(HGCalLayerClusterProducerEMGPU);

HGCalLayerClusterProducerEMGPU::HGCalLayerClusterProducerEMGPU(const edm::ParameterSet& ps)
  : mDc(ps.getParameter<double>("dc")),
    mKappa(ps.getParameter<double>("kappa")),
    mEcut(ps.getParameter<double>("ecut")),
    mOutlierDeltaFactor(ps.getParameter<double>("outlierDeltaFactor")),
    gpuPositionsTok_(esConsumes<HeterogeneousHGCalPositionsConditions, HeterogeneousHGCalPositionsConditionsRecord>()),
    InEEToken{consumes<cms::cuda::Product<HGCRecHitGPUProduct>>(ps.getParameter<edm::InputTag>("EMInputRecHitsGPU"))},
    OutEEHitsToken{produces<cms::cuda::Product<HGCCLUEGPUHitsProduct>>("Hits")},
    OutEEClustersToken{produces<cms::cuda::Product<HGCCLUEGPUClustersProduct>>("Clusters")}
{}

// void HGCalLayerClusterProducerEMGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   // hgcalLayerClusters
//   edm::ParameterSetDescription desc;

//   desc.add<edm::InputTag>("HGCEEInputGPU", edm::InputTag("EERecHitGPUProd"));
//   descriptions.add("hgcalLayerClustersGPU", desc);
// }

void HGCalLayerClusterProducerEMGPU::acquire(edm::Event const& event,
					   edm::EventSetup const& es,
					   edm::WaitingTaskWithArenaHolder w) {
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w), ctxState_};
  const auto& eeHits = ctx.get(event, InEEToken);
  const unsigned nhits(eeHits.nHits());
  unsigned nclusters(clue_gpu::maxNSeeds);
  unsigned nlayersEM(28); //upper value; we could use the geometry + rechittools, but it seems an overkill just for a single number
  
  mCLUEHits = HGCCLUEGPUHitsProduct(nhits, ctx.stream());
  mCLUEClusters = HGCCLUEGPUClustersProduct(nclusters, ctx.stream());
  
  //retrieve HGCAL positions conditions data
  auto hPosConds = es.getHandle(gpuPositionsTok_);
  const auto* gpuPositionsConds = hPosConds->getHeterogeneousConditionsESProductAsync(ctx.stream());

  //CLUE
  mAlgo = std::make_unique<HGCalCLUEAlgoGPUEM>(mDc, mKappa, mEcut, mOutlierDeltaFactor,
  					       mCLUEHits.get(), mCLUEClusters.get());

  mAlgo->populate(eeHits.get(), gpuPositionsConds, ctx.stream());
  mAlgo->make_clusters(ctx.stream());

  //Clusters
  mAlgo->get_clusters(nlayersEM, ctx.stream());
}

void HGCalLayerClusterProducerEMGPU::produce(edm::Event& event,
					     const edm::EventSetup& es) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};
  ctx.emplace(event, OutEEHitsToken, std::move(mCLUEHits));
  ctx.emplace(event, OutEEClustersToken, std::move(mCLUEClusters));
}

#endif  //__RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducerEMGPU_H__
