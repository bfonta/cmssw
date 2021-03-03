#include <iostream>

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEFCellPositionsConditions.h"
#include "CondFormats/DataRecord/interface/HeterogeneousHGCalHEFCellPositionsConditionsRecord.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalCellPositions.h"


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalCellPositions.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEFCellPositionsConditions.h"
#include "Geometry/Records/src/IdealGeometryRecord.cc"

class BrunoDummyProducer : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit BrunoDummyProducer(const edm::ParameterSet &ps);
  ~BrunoDummyProducer() override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<HGChefUncalibratedRecHitCollection> Tok_;
  edm::ESGetToken<HeterogeneousHGCalHEFCellPositionsConditions, HeterogeneousHGCalHEFCellPositionsConditionsRecord> esToken_;
  edm::EDPutTokenT<HGChefRecHitCollection> putTok_;
};

BrunoDummyProducer::BrunoDummyProducer(const edm::ParameterSet& ps)
  : Tok_{consumes<HGCUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("Tok"))},
    esToken_(esConsumes<HeterogeneousHGCalHEFCellPositionsConditions, HeterogeneousHGCalHEFCellPositionsConditionsRecord>()) {
  putTok_ = produces<HGChefRecHitCollection>();
}

BrunoDummyProducer::~BrunoDummyProducer() {}

void BrunoDummyProducer::beginRun(edm::Run const&, edm::EventSetup const& setup) {}

void BrunoDummyProducer::acquire(edm::Event const& event,
			    edm::EventSetup const& setup,
			    edm::WaitingTaskWithArenaHolder w) {
  std::cout << "acquire() called" << std::endl;
  const auto& hits = event.get(Tok_);
  hits.size(); //do something to avoid warnings
  
  cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(w)};
  auto handle = setup.getHandle(esToken_);

  const auto* d_conds = handle->getHeterogeneousConditionsESProductAsync(ctx.stream());

  KernelManagerHGCalCellPositions kernel_manager( 1 );
  kernel_manager.test_cell_positions(2416969935, d_conds);
}

void BrunoDummyProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  std::cout << "produce() called" << std::endl;
  std::unique_ptr<HGChefRecHitCollection> rechits_ = std::make_unique<HGCRecHitCollection>();
  event.put(std::move(rechits_));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BrunoDummyProducer);
