#ifndef RecoLocalCalo_HGCalRecProducers_HeterogeneousHGCalHEFCellPositionsFiller_h
#define RecoLocalCalo_HGCalRecProducers_HeterogeneousHGCalHEFCellPositionsFiller_h

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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEFCellPositionsConditions.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalProducerMemoryWrapper.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "CondFormats/DataRecord/interface/HeterogeneousHGCalHEFCellPositionsConditionsRecord.h"

class HeterogeneousHGCalHEFCellPositionsFiller: public edm::ESProducer 
{
 public:
  explicit HeterogeneousHGCalHEFCellPositionsFiller(const edm::ParameterSet&);
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
