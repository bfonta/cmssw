#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEM.h"

HGCalCLUEAlgoGPUEM::HGCalCLUEAlgoGPUEM(float dc, float kappa, float ecut, float outlierDeltaFactor,
				       const HGCCLUESoA& hits_soa, uint32_t nhits)
  : HGCalCLUEAlgoGPUBase(dc, kappa, ecut, outlierDeltaFactor, hits_soa, nhits)
{}

HGCalCLUEAlgoGPUEM::HGCalCLUEAlgoGPUEM(const HGCCLUESoA& clueSoAHost,
				       const ConstHGCCLUESoA& clueSoADev, uint32_t nhits)
  : HGCalCLUEAlgoGPUBase(clueSoAHost, clueSoADev, nhits)
{}

void HGCalCLUEAlgoGPUEM::copy_tohost(const cudaStream_t& s) {
  HGCalCLUEAlgoGPUBase::copy_tohost(s);
}  

void HGCalCLUEAlgoGPUEM::set_input_SoA_layout(const cudaStream_t &stream) {
  const std::array<uint32_t, clue_gpu::ntypes_hgcclue_inemsoa> sizes_ = {
		{clue_gpu::float_hgcclue_inemsoa * sizeof(float),
		 clue_gpu::int32_hgcclue_inemsoa * sizeof(int32_t)}
  };
  const uint32_t size_tot = std::accumulate(sizes_.begin(), sizes_.end(), 0);
  mMem = allocate_soa_memory_block(size_tot, stream);
  
  //set input SoA memory view
  mDevPoints.x          = reinterpret_cast<float *>(mMem.get());
  mDevPoints.y          = mDevPoints.x      + mPad;
  mDevPoints.energy     = mDevPoints.y      + mPad;
  mDevPoints.sigmaNoise = mDevPoints.energy + mPad;
  mDevPoints.layer      = reinterpret_cast<int32_t *>(mDevPoints.sigmaNoise + mPad);

  mDevPoints.pad = mPad;
}
				  
void HGCalCLUEAlgoGPUEM::populate(const ConstHGCRecHitSoA& hits,
				  const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds,
				  const cudaStream_t& stream) {
  set_input_SoA_layout(stream);
  set_memory();

  const dim3 blockSize(mNThreadsEM,1,1);
  const dim3 gridSize( calculate_block_multiplicity(mNHits, blockSize.x), 1, 1 );

  kernel_fill_input_soa<<<gridSize,blockSize,0,stream>>>(hits, mDevPoints, conds, mEcut);

  cudaCheck( cudaStreamSynchronize(stream) );
}

void HGCalCLUEAlgoGPUEM::make_clusters(const cudaStream_t &stream) {
  const dim3 blockSize(mNThreadsEM,1,1);
  //const dim3 gridSize( calculate_block_multiplicity(mNHits, blockSize.x), 1, 1 );
  const dim3 gridSize( 1, 1, 1 );

  ////////////////////////////////////////////
  // calculate rho, delta and find seeds
  // 1 point per thread
  ////////////////////////////////////////////
  kernel_compute_histogram<<<gridSize,blockSize,0,stream>>>(mDevHist, mDevPoints, mNHits);

  kernel_calculate_density<<<gridSize,blockSize,0,stream>>>(mDevHist, mDevPoints, mCLUESoA,
							    mDc, mNHits);

  kernel_calculate_distanceToHigher<<<gridSize,blockSize,0,stream>>>(mDevHist, mDevPoints, mCLUESoA,
								     mOutlierDeltaFactor, mDc,
								     mNHits);

  kernel_find_clusters<<<gridSize,blockSize,0,stream>>>(mDevSeeds, mDevFollowers,
							mDevPoints, mCLUESoA,
							mOutlierDeltaFactor, mDc, mKappa,
							mNHits);
  
  ////////////////////////////////////////////
  // assign clusters
  // 1 point per seeds
  ////////////////////////////////////////////
  // const dim3 gridSize_nseeds( calculate_block_multiplicity(clue_gpu::maxNSeeds, blockSize.x), 1, 1 );
  const dim3 gridSize_nseeds( 1, 1, 1 );
  kernel_assign_clusters<<<gridSize_nseeds,blockSize,0,stream>>>(mDevSeeds, mDevFollowers, mCLUESoA);
}
