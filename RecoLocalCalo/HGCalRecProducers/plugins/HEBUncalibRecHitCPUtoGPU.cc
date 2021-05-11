#include <string>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitDevice.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitHost.h"

class HEBUncalibRecHitCPUtoGPU : public edm::stream::EDProducer<> {
public:
  explicit HEBUncalibRecHitCPUtoGPU(const edm::ParameterSet &ps);
  ~HEBUncalibRecHitCPUtoGPU() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<HGChebUncalibratedRecHitCollection> uncalibRecHitCPUToken_;
  edm::EDPutTokenT<cms::cuda::Product<HGCUncalibRecHitDevice>> uncalibRecHitGPUToken_;

  //constants
  HGChebUncalibRecHitConstantData cdata_;
  HGCConstantVectorData vdata_;

  //conditions (geometry, topology, ...)
  std::unique_ptr<hgcal::RecHitTools> tools_;

  //
  HGCUncalibRecHitDevice d_uncalib_;
  HGCUncalibRecHitHost<HGChebUncalibratedRecHitCollection> h_uncalib_;
};

HEBUncalibRecHitCPUtoGPU::HEBUncalibRecHitCPUtoGPU(const edm::ParameterSet &ps)
    : uncalibRecHitCPUToken_{consumes<HGCUncalibratedRecHitCollection>(
          ps.getParameter<edm::InputTag>("HGCHEBUncalibRecHitsTok"))},
      uncalibRecHitGPUToken_{produces<cms::cuda::Product<HGCUncalibRecHitDevice>>()} {
  tools_ = std::make_unique<hgcal::RecHitTools>();
}

HEBUncalibRecHitCPUtoGPU::~HEBUncalibRecHitCPUtoGPU() {}

void HEBUncalibRecHitCPUtoGPU::beginRun(edm::Run const &, edm::EventSetup const &setup) {}

void HEBUncalibRecHitCPUtoGPU::produce(edm::Event &event, const edm::EventSetup &setup) {
  cms::cuda::ScopedContextProduce ctx{event.streamID()};
  
  const auto &hits = event.get(uncalibRecHitCPUToken_);
  const unsigned nhits(hits.size());

  if (nhits == 0)
    edm::LogError("HEBUncalibRecHitCPUtoGPU") << "WARNING: no input hits!";

  d_uncalib_ = HGCUncalibRecHitDevice(nhits, ctx.stream());
  h_uncalib_ = HGCUncalibRecHitHost<HGChebUncalibratedRecHitCollection>(nhits, hits, ctx.stream());

  KernelManagerHGCalRecHit km(h_uncalib_.get(), d_uncalib_.get());
  km.transfer_soa_to_device(ctx.stream());

  ctx.emplace(event, uncalibRecHitGPUToken_, std::move(d_uncalib_));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HEBUncalibRecHitCPUtoGPU);
