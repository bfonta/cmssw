#ifndef RecoLocalCalo_HGCalRecProducers_HeterogeneousHGCalHEFCellPositionsFiller_h
#define RecoLocalCalo_HGCalRecProducers_HeterogeneousHGCalHEFCellPositionsFiller_h

#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "HeterogeneousHGCalHEFCellPositionsConditions.h"
#include "HeterogeneousHGCalProducerMemoryWrapper.h"
#include "KernelManagerHGCalRecHit.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

class HeterogeneousHGCalHEFCellPositionsFiller: public edm::ESProducer 
{
 public:
  explicit HeterogeneousHGCalHEFCellPositionsFiller(const edm::ParameterSet& ps);
  ~HeterogeneousHGCalHEFCellPositionsFiller() override;
  std::unique_ptr<HeterogeneousHGCalHEFCellPositionsConditions> produce(const HeterogeneousHGCalHEFCellPositionsConditionsRecord&);

 private:
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geometryToken_;
  
  //cms::cuda::ContextState ctxState_;

  //conditions (geometry, topology, ...)
  void set_conditions_();

  const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* d_conds = nullptr;
  hgcal_conditions::positions::HGCalPositionsMapping* posmap_;

  const HGCalDDDConstants* ddd_ = nullptr;
  const HGCalParameters* params_ = nullptr;
};

#endif //RecoLocalCalo_HGCalRecProducers_HeterogeneousHGCalHEFCellPositionsFiller_h
