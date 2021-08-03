#include <cuda.h>
#include <cuda_runtime.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"

HGCalCLUEAlgoGPUBase::HGCalCLUEAlgoGPUBase(float pDc, float pKappa, float pEcut,
					   float pOutlierDeltaFactor,
					   const HGCCLUEHitsSoA& pCLUEHitsSoA,
					   const HGCCLUEHitsSoA& pCLUEHitsSoA,
					   uint32_t nhits, uint32_t nclusters)
  : mDc(pDc), mKappa(pKappa), mEcut(pEcut), mOutlierDeltaFactor(pOutlierDeltaFactor), mCLUEHitsSoA(pCLUEHitsSoA), mCLUEClustersSoA(pCLUEClustersSoA),
    mNHits(nhits), mNClusters(nclusters)
{
  mPadHits = calculate_padding(mNHits);
  mPadClusters = calculate_padding(mNClusters);

  cudaMalloc(&mDevHist, sizeof(LayerTilesGPU) * NLAYERS);
  cudaMalloc(&mDevSeeds, sizeof(cms::cuda::VecArray<int,clue_gpu::maxNSeeds>) );
  cudaMalloc(&mDevFollowers, sizeof(cms::cuda::VecArray<int,clue_gpu::maxNFollowers>)*mNHits);

  was_memory_allocated = true;
}

HGCalCLUEAlgoGPUBase::HGCalCLUEAlgoGPUBase(const HGCCLUEHitsSoA& pCLUEHitsSoAHost, const ConstHGCCLUEHitsSoA& pCLUEHitsSoADev,
					   const HGCCLUEClustersSoA& pCLUEClustersSoAHost, const ConstHGCCLUEClustersSoA& pCLUEClustersSoADev,
					   uint32_t nhits)
  : mCLUEHitsSoAHost(pCLUESoAHost), mCLUEHitsSoADev(pCLUESoADev), mCLUEClustersSoAHost(pCLUESoAHost), mCLUEClustersSoADev(pCLUESoADev), mNHits(nhits)
{
  mPadHits = calculate_padding(mNHits);
  mPadClusters = calculate_padding(mNClusters);

  was_memory_allocated = false;
}

HGCalCLUEAlgoGPUBase::~HGCalCLUEAlgoGPUBase() {
  if(was_memory_allocated)
    free_device();
}
    
void HGCalCLUEAlgoGPUBase::free_device() {
  // algorithm internal variables
  cudaFree(mDevHist);
  cudaFree(mDevSeeds);
  cudaFree(mDevFollowers);
}

void HGCalCLUEAlgoGPUBase::set_memory() {
  // condense into single memset??
  cudaMemset(mCLUEHitsSoA.rho,           0x00, sizeof(float)*mPadHits);
  cudaMemset(mCLUEHitsSoA.delta,         0x00, sizeof(float)*mPadHits);
  cudaMemset(mCLUEHitsSoA.nearestHigher, 0x00, sizeof(int)*mPadHits);
  cudaMemset(mCLUEHitsSoA.clusterIndex,  0x00, sizeof(int)*mPadHits);
  cudaMemset(mCLUEHitsSoA.isSeed,        0x00, sizeof(int)*mPadHits);
  
  // algorithm internal variables
  cudaMemset(mDevHist, 0x00, sizeof(LayerTilesGPU) * NLAYERS);
  cudaMemset(mDevSeeds, 0x00, sizeof(GPU::VecArray<int,clue_gpu::maxNSeeds>));
  cudaMemset(mDevFollowers, 0x00, sizeof(GPU::VecArray<int,clue_gpu::maxNFollowers>)*mNHits);
}

void HGCalCLUEAlgoGPUBase::copy_tohost(const cudaStream_t& stream) {
  //the original standalone version transferred only the cluster index
  cudaMemcpyAsync(mCLUEHitsSoAHost.rho, mCLUEHitsSoADev.rho,
		  mPadHits*mCLUEHitsSoAHost.nbytes, cudaMemcpyDeviceToHost, stream);

  cudaMemcpyAsync(mCLUEClustersSoAHost.energy, mCLUEClustersSoADev.energy,
		  mPadClusters*mCLUEClustersSoAHost.nbytes, cudaMemcpyDeviceToHost, stream);
}

uint32_t HGCalCLUEAlgoGPUBase::calculate_padding(uint32_t n) {
  //align to warp boundary (assumption: warpSize = 32)
  return ((n - 1) / 32 + 1) * 32;
}

float HGCalCLUEAlgoGPUBase::calculate_block_multiplicity(unsigned nelements, unsigned nthreads) {
  return ceil(nelements/static_cast<float>(nthreads));
}

cms::cuda::device::unique_ptr<std::byte[]>
HGCalCLUEAlgoGPUBase::allocate_soa_memory_block(uint32_t st, const cudaStream_t &stream) {
  return cms::cuda::make_device_unique<std::byte[]>(mPadHits * st, stream);
}
