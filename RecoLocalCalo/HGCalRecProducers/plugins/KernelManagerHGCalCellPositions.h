#ifndef RecoLocalCalo_HGCalESProducers_KernelManagerHGCalCellPositions_h
#define RecoLocalCalo_HGCalESProducers_KernelManagerHGCalCellPositions_h

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cuh"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

#include <cuda.h>
#include <cuda_runtime.h>

void fill_positions(unsigned, unsigned,
		    const hgcal_conditions::HeterogeneousPositionsConditionsESProduct*,
		    const cudaStream_t&);

void test_cell_positions(unsigned, unsigned, unsigned,
			 const hgcal_conditions::HeterogeneousPositionsConditionsESProduct*,
			 const cudaStream_t&);

#endif  //RecoLocalCalo_HGCalESProducers_KernelManagerHGCalCellPositions_h
