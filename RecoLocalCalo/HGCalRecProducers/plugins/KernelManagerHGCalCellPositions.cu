#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalCellPositions.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cuh"

void fill_positions(unsigned nthreads, unsigned nblocks,
		    const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* d_conds,
		    const cudaStream_t& stream) {

  dim3 nt(nthreads);
  dim3 nb(nblocks);
  fill_positions_from_detids<<<nb, nt, 0, stream>>>(d_conds);
}

void test_cell_positions(unsigned nthreads, unsigned nblocks,
			 unsigned id, const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* d_conds,
			 const cudaStream_t& stream) {
  dim3 nt(nthreads);
  dim3 nb(nblocks);
  test<<<nb, nt, 0, stream>>>(id, d_conds);
}
