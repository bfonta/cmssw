#include <cuda.h>
#include <cuda_runtime.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"

HGCalCLUEAlgoGPUBase::HGCalCLUEAlgoGPUBase(float pDc, float pKappa, float pEcut,
					   float pOutlierDeltaFactor,
					   const HGCCLUESoA& pCLUESoA,
					   uint32_t nhits)
  : mDc(pDc), mKappa(pKappa), mEcut(pEcut), mOutlierDeltaFactor(pOutlierDeltaFactor), mCLUESoA(pCLUESoA), mNHits(nhits)
{
  mPad = calculate_padding(mNHits);
}

HGCalCLUEAlgoGPUBase::HGCalCLUEAlgoGPUBase(const HGCCLUESoA& pCLUESoAHost,
					   const ConstHGCCLUESoA& pCLUESoADev,
					   uint32_t nhits)
  : mCLUESoAHost(pCLUESoAHost), mCLUESoADev(pCLUESoADev), mNHits(nhits)
{
  mPad = calculate_padding(mNHits);
}

HGCalCLUEAlgoGPUBase::~HGCalCLUEAlgoGPUBase() { free_device(); }
    
void HGCalCLUEAlgoGPUBase::free_device() {
  // algorithm internal variables
  cudaFree(mDevHist);
  cudaFree(mDevSeeds);
  cudaFree(mDevFollowers);
}

void HGCalCLUEAlgoGPUBase::allocate_common_memory_blocks() {
  cudaMalloc(&mDevHist, sizeof(LayerTilesGPU) * NLAYERS);
  cudaMalloc(&mDevSeeds, sizeof(cms::cuda::VecArray<int,clue_gpu::maxNSeeds>) );
  cudaMalloc(&mDevFollowers, sizeof(cms::cuda::VecArray<int,clue_gpu::maxNFollowers>)*mNHits);
}

void HGCalCLUEAlgoGPUBase::set_memory() {
  // condense into single memset??
  cudaMemset(mCLUESoA.rho,           0x00, sizeof(float)*mPad);
  cudaMemset(mCLUESoA.delta,         0x00, sizeof(float)*mPad);
  cudaMemset(mCLUESoA.nearestHigher, 0x00, sizeof(int)*mPad);
  cudaMemset(mCLUESoA.clusterIndex,  0x00, sizeof(int)*mPad);
  cudaMemset(mCLUESoA.isSeed,        0x00, sizeof(int)*mPad);
  
  // algorithm internal variables
  cudaMemset(mDevHist, 0x00, sizeof(LayerTilesGPU) * NLAYERS);
  cudaMemset(mDevSeeds, 0x00, sizeof(GPU::VecArray<int,clue_gpu::maxNSeeds>));
  cudaMemset(mDevFollowers, 0x00, sizeof(GPU::VecArray<int,clue_gpu::maxNFollowers>)*mNHits);
}

void HGCalCLUEAlgoGPUBase::copy_tohost(const cudaStream_t& stream) {
  //the original standalone version transferred only the cluster index
  cudaMemcpyAsync(mCLUESoAHost.rho, mCLUESoADev.rho, mPad*mCLUESoAHost.nbytes, cudaMemcpyDeviceToHost);
}

uint32_t HGCalCLUEAlgoGPUBase::calculate_padding(uint32_t nhits) {
  //align to warp boundary (assumption: warpSize = 32)
  return ((nhits - 1) / 32 + 1) * 32;
}

float HGCalCLUEAlgoGPUBase::calculate_block_multiplicity(unsigned nelements, unsigned nthreads) {
  return ceil(nelements/static_cast<float>(nthreads));
}

cms::cuda::device::unique_ptr<std::byte[]>
HGCalCLUEAlgoGPUBase::allocate_soa_memory_block(uint32_t st, const cudaStream_t &stream) {
  return cms::cuda::make_device_unique<std::byte[]>(mPad * st, stream);
}
