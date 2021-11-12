#ifndef HGCalCLUEAlgoGPUBase_h
#define HGCalCLUEAlgoGPUBase_h

#include <cuda_runtime.h>
#include <cuda.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HeterogeneousHGCalLayerTiles.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

namespace clue_gpu {
  static const int maxNSeeds = 1000000; 
  static const int maxNFollowers = 500;
  static const int localStackSizePerSeed = 500;

  //number of float pointers in the CLUE EM SoA
  constexpr unsigned float_hgcclue_inemsoa = 4;
  constexpr unsigned int32_hgcclue_inemsoa = 1;
  constexpr unsigned uint32_hgcclue_inemsoa = 1;
  //number of different pointer types in the CLUE EM SoA
  constexpr unsigned ntypes_hgcclue_inemsoa = 3;

  class HGCCLUEInputSoAEM {
  public:
    float *x; //x position of the calibrated rechit
    float *y; //y position of the calibrated rechit
    float *energy; //calibrated energy of the rechit
    float *sigmaNoise; //calibrated noise of the rechit cell
    int32_t *layer; //layer position of the calibrated rechit
    uint32_t *id; //detid position of the calibrated rechit
    
    uint32_t pad; //pad of memory block (used for warp alignment, slightly larger than 'nhits_')
  };
}

class HGCalCLUEAlgoGPUBase {
public:
  HGCalCLUEAlgoGPUBase(float, float, float, float,
		       const HGCCLUEHitsSoA&, const HGCCLUEClustersSoA&);
  
  HGCalCLUEAlgoGPUBase(unsigned, unsigned,
		       const HGCCLUEHitsSoA&, const ConstHGCCLUEHitsSoA&,
		       const HGCCLUEClustersSoA&, const ConstHGCCLUEClustersSoA&);
  
protected:
  //when using polymorphism the base destructor should be instead
  //made virtual to avoid not calling the derived destructor
  ~HGCalCLUEAlgoGPUBase();

  float mDc, mKappa, mEcut, mOutlierDeltaFactor;
  uint32_t mNHits, mNClusters;
  uint32_t mPadHits, mPadClusters;
  cms::cuda::device::unique_ptr<std::byte[]> mMem;

  HGCCLUEHitsSoA mCLUEHitsSoAHost, mCLUEHitsSoA;
  ConstHGCCLUEHitsSoA mCLUEHitsSoADev;
  HGCCLUEClustersSoA mCLUEClustersSoAHost, mCLUEClustersSoA;
  ConstHGCCLUEClustersSoA mCLUEClustersSoADev;

  HeterogeneousHGCalLayerTiles *mDevHist;
  cms::cuda::VecArray<int,clue_gpu::maxNSeeds> *mDevSeeds;
  cms::cuda::VecArray<int,clue_gpu::maxNFollowers> *mDevFollowers;

  uint32_t calculate_padding(uint32_t);
  float calculate_block_multiplicity(unsigned, unsigned);
  void set_memory();
  void copy_tohost(const cudaStream_t&);
  cms::cuda::device::unique_ptr<std::byte[]> allocate_soa_memory_block(uint32_t,
								       const cudaStream_t &);
  void free_device();

private:
  virtual void populate(const ConstHGCRecHitSoA&,
			const hgcal_conditions::HeterogeneousPositionsConditionsESProduct*,
			const cudaStream_t&) = 0;
  virtual void make_clusters(const cudaStream_t&) = 0;
  virtual void get_clusters(const cudaStream_t&) = 0;

  bool was_memory_allocated;
};

#endif // HGCalCLUEAlgoGPUBase_h
