#ifndef CUDADataFormats_HGCal_HeterogeneousHGCalHEFCellPositionsConditionsRecord_h
#define CUDADataFormats_HGCal_HeterogeneousHGCalHEFCellPositionsConditionsRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "FWCore/Utilities/interface/mplVector.h"

class HeterogeneousHGCalHEFCellPositionsConditionsRecord
    : public edm::eventsetup::DependentRecordImplementation<HeterogeneousHGCalHEFCellPositionsConditionsRecord, edm::mpl::Vector<IdealGeometryRecord>> {};

#endif //CUDADataFormats_HGCal_HeterogeneousHGCalHEFCellPositionsConditionsRecord_h
