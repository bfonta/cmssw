#ifndef HGCalCLUEAlgoGPUBase_h
#define HGCalCLUEAlgoGPUBase_h

#include <cuda_runtime.h>
#include <cuda.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/LayerTilesGPU.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

namespace clue_gpu {
  static const int maxNSeeds = 100000; 
  static const int maxNFollowers = 2000; 
  static const int localStackSizePerSeed = 2000;
  static const float unphysicalEnergy = -1.f;

  //number of float pointers in the CLUE EM SoA
  constexpr unsigned float_hgcclue_inemsoa = 4;
  constexpr unsigned int32_hgcclue_inemsoa = 1;
  //number of different pointer types in the CLUE EM SoA
  constexpr unsigned ntypes_hgcclue_inemsoa = 2;

  class HGCCLUEInputSoAEM {
  public:
    float *x; //x position of the calibrated rechit
    float *y; //y position of the calibrated rechit
    float *energy; //calibrated energy of the rechit
    float *sigmaNoise; //calibrated noise of the rechit cell
    int32_t *layer; //layer position of the calibrated rechit
    
    uint32_t pad; //pad of memory block (used for warp alignment, slightly larger than 'nhits_')
  };
}

class HGCalCLUEAlgoGPUBase {
public:
  HGCalCLUEAlgoGPUBase(float, float, float, float,
		       const HGCCLUESoA&);

protected:
  //when using polymorphism the base destructor should be instead
  //made virtual to avoid not calling the derived destructor
  ~HGCalCLUEAlgoGPUBase();

  float mDc, mKappa, mEcut, mOutlierDeltaFactor;
  cms::cuda::device::unique_ptr<std::byte[]> mMem;
  HGCCLUESoA mCLUESoA;
  LayerTilesGPU *mDevHist;
  cms::cuda::VecArray<int,clue_gpu::maxNSeeds> *mDevSeeds;
  cms::cuda::VecArray<int,clue_gpu::maxNFollowers> *mDevFollowers;

  uint32_t calculate_padding(uint32_t);
  float calculate_block_multiplicity(unsigned, unsigned);
  void allocate_common_memory_blocks(uint32_t);
  cms::cuda::device::unique_ptr<std::byte[]> allocate_soa_memory_block(uint32_t,
								       uint32_t,
								       const cudaStream_t &);
  void free_device();

private:
  virtual void populate(const ConstHGCRecHitSoA&,
			const hgcal_conditions::HeterogeneousPositionsConditionsESProduct*,
			const cudaStream_t&) = 0;
  virtual void make_clusters(const unsigned, const cudaStream_t&) = 0;
};

#endif // HGCalCLUEAlgoGPUBase_h
