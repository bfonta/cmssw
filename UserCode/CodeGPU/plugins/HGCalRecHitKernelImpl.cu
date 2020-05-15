#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "HGCalRecHitKernelImpl.cuh"
#include "HGCalDetIdTools.h"

__device__ 
double get_weight_from_layer(const int& padding, const int& layer, double*& sd)
{
  return sd[padding + layer];
}

__device__
void make_rechit(unsigned int tid, HGCRecHitSoA& dst_soa, HGCUncalibratedRecHitSoA& src_soa, const bool &heb_flag, 
		 const double &weight, const double &rcorr, const double &cce_correction, const double &sigmaNoiseGeV, float *& sf)
{
  dst_soa.id_[tid] = src_soa.id_[tid];
  dst_soa.energy_[tid] = src_soa.amplitude_[tid] * weight * 0.001f;
  if(!heb_flag)
    dst_soa.energy_[tid] *=  __fdividef(rcorr, cce_correction);
  dst_soa.time_[tid] = src_soa.jitter_[tid];
  dst_soa.flagBits_[tid] |= (0x1 << HGCRecHit::kGood);
  float son = __fdividef( dst_soa.energy_[tid], sigmaNoiseGeV);
  float son_norm = fminf(32.f, son) / 32.f * ((1 << 8)-1);
  long int son_round = lroundf( son_norm );
  dst_soa.son_[tid] = static_cast<uint8_t>( son_round );

  if(heb_flag==0)
    {
      //get time resolution
      float max = fmaxf(son, sf[0]); //this max trick avoids if...elseif...else condition
      float aterm = sf[2];
      float cterm = sf[3];
      dst_soa.timeError_[tid] = sqrt( __fdividef(aterm,max)*__fdividef(aterm,max) + cterm*cterm );
    }
  else
    dst_soa.timeError_[tid] = -1;
}

__device__ 
double get_thickness_correction(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL];
}
__device__ 
double get_thickness_correction(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL];
}

__device__
double get_noise(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}
__device__
double get_noise(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}

__device__
double get_cce_correction(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}
__device__
double get_cce_correction(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}

__device__ 
double get_fCPerMIP(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained from the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}
__device__ 
double get_fCPerMIP(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained from the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}

__device__ 
void set_shared_memory(const int& tid, double*& sd, float*& sf, int*& si, const HGCeeUncalibratedRecHitConstantData& cdata, const int& size1, const int& size2, const int& size3, const int& size4, const int& size5, const int& size6)
{
  const int initial_pad = 2;
  if(tid == 0)
    {
      sd[0] = cdata.hgcEE_keV2DIGI_;
      sd[1] = cdata.hgceeUncalib2GeV_;
      for(unsigned int i=initial_pad; i<size1; ++i)
	sd[i] = cdata.hgcEE_fCPerMIP_[i-initial_pad];
      for(unsigned int i=size1; i<size2; ++i)
	sd[i] = cdata.hgcEE_cce_[i-size1];
      for(unsigned int i=size2; i<size3; ++i)
	sd[i] = cdata.hgcEE_noise_fC_[i-size2];  
      for(unsigned int i=size3; i<size4; ++i)
	sd[i] = cdata.rcorr_[i-size3];
      for(unsigned int i=size4; i<size5; ++i)
	sd[i] = cdata.weights_[i-size4];
      for(unsigned int i=size5; i<size6; ++i)
	si[i-size5] = cdata.waferTypeL_[i-size5];
      sf[0] = (cdata.xmin_ > 0) ? cdata.xmin_ : 0.1;
      sf[1] = cdata.xmax_;
      sf[2] = cdata.aterm_;
      sf[3] = cdata.cterm_;
    }
}

__device__ 
void set_shared_memory(const int& tid, double*& sd, float*& sf, uint32_t*& su, int*& si, const HGChefUncalibratedRecHitConstantData& cdata, const int& size1, const int& size2, const int& size3, const int& size4, const int& size5, const int& size6)
{
  const int initial_pad = 2;
  if(tid == 0)
    {
      sd[0] = cdata.hgcHEF_keV2DIGI_;
      sd[1] = cdata.hgchefUncalib2GeV_;
      for(unsigned int i=initial_pad; i<size1; ++i)
	sd[i] = cdata.hgcHEF_fCPerMIP_[i-initial_pad];
      for(unsigned int i=size1; i<size2; ++i)
	sd[i] = cdata.hgcHEF_cce_[i-size1];
      for(unsigned int i=size2; i<size3; ++i)
	sd[i] = cdata.hgcHEF_noise_fC_[i-size2];  
      for(unsigned int i=size3; i<size4; ++i)
	sd[i] = cdata.rcorr_[i-size3];
      for(unsigned int i=size4; i<size5; ++i)
	sd[i] = cdata.weights_[i-size4];
      for(unsigned int i=size5; i<size6; ++i)
	si[i-size5] = cdata.waferTypeL_[i-size5];
      sf[0] = (cdata.xmin_ > 0) ? cdata.xmin_ : 0.1;
      sf[1] = cdata.xmax_;
      sf[2] = cdata.aterm_;
      sf[3] = cdata.cterm_;
      su[0] = cdata.fhOffset_;
    }
}

__device__ 
void set_shared_memory(const int& tid, double*& sd, float*& sf, uint32_t*& su, const HGChebUncalibratedRecHitConstantData& cdata, const int& size1)
{
  const int initial_pad = 3;
  if(tid == 0)
    {
      sd[0] = cdata.hgcHEB_keV2DIGI_;
      sd[1] = cdata.hgchebUncalib2GeV_;
      sd[2] = cdata.hgcHEB_noise_MIP_;
      for(unsigned int i=initial_pad; i<size1; ++i)
	sd[i] = cdata.weights_[i-initial_pad];
      su[0] = cdata.bhOffset_;
    }
}

__global__
void ee_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGCeeUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void hef_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChefUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void heb_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChebUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void ee_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGCeeUncalibratedRecHitConstantData cdata, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int size1 = cdata.s_hgcEE_fCPerMIP_ + 2;
  int size2 = cdata.s_hgcEE_cce_      + size1;
  int size3 = cdata.s_hgcEE_noise_fC_ + size2;
  int size4 = cdata.s_rcorr_          + size3; 
  int size5 = cdata.s_weights_        + size4; 
  int size6 = cdata.s_waferTypeL_     + size5; 

  extern __shared__ double s[];
  double   *sd = s;
  float    *sf = (float*)   (sd + cdata.ndelem_);
  int      *si = (int*)     (sf + cdata.nfelem_);
  set_shared_memory(threadIdx.x, sd, sf, si, cdata, size1, size2, size3, size4, size5, size6);
  __syncthreads();

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double weight = get_weight_from_layer(size4, src_soa.layer_[i], sd);
      double rcorr = get_thickness_correction(size3, sd, cdata);
      double noise = get_noise(size2, sd, cdata);
      double cce_correction = get_cce_correction(size1, sd, cdata);
      double fCPerMIP = get_fCPerMIP(2, sd, cdata);
      double sigmaNoiseGeV = 1e-3 * weight * rcorr * __fdividef( noise, fCPerMIP );
      make_rechit(i, dst_soa, src_soa, false, weight, rcorr, cce_correction, sigmaNoiseGeV, sf);
    }
}

__global__
void hef_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChefUncalibratedRecHitConstantData cdata, const HeterogeneousHEFConditionsESProduct* conds, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  HeterogeneousHGCalDetId detid(src_soa.id_[0]);
  printf("%d, %d\n", conds->ddd.waferTypeL[0], detid.layer());

  int size1 = cdata.s_hgcHEF_fCPerMIP_ + 2;
  int size2 = cdata.s_hgcHEF_cce_      + size1;
  int size3 = cdata.s_hgcHEF_noise_fC_ + size2;
  int size4 = cdata.s_rcorr_           + size3; 
  int size5 = cdata.s_weights_         + size4; 
  int size6 = cdata.s_waferTypeL_      + size5;

  extern __shared__ double s[];
  double   *sd = s;
  float    *sf = (float*)   (sd + cdata.ndelem_);
  uint32_t *su = (uint32_t*)(sf + cdata.nfelem_);
  int      *si = (int*)     (su + cdata.nuelem_);
  set_shared_memory(threadIdx.x, sd, sf, su, si, cdata, size1, size2, size3, size4, size5, size6);
  __syncthreads();

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double weight = get_weight_from_layer(size4, src_soa.layer_[i], sd);
      double rcorr = get_thickness_correction(size3, sd, cdata);
      double noise = get_noise(size2, sd, cdata);
      double cce_correction = get_cce_correction(size1, sd, cdata);
      double fCPerMIP = get_fCPerMIP(2, sd, cdata);
      double sigmaNoiseGeV = 1e-3 * weight * rcorr * __fdividef( noise,  fCPerMIP );
      make_rechit(i, dst_soa, src_soa, false, weight, rcorr, cce_correction, sigmaNoiseGeV, sf);
    }
}

__global__
void heb_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChebUncalibratedRecHitConstantData cdata, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int size1 = cdata.s_weights_ + 3; 

  extern __shared__ double s[];
  double   *sd = s;
  float    *sf = (float*)   (sd + cdata.ndelem_);
  uint32_t *su = (uint32_t*)(sf + cdata.nfelem_);
  set_shared_memory(threadIdx.x, sd, sf, su, cdata, size1);
  __syncthreads();

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double weight = get_weight_from_layer(3, src_soa.layer_[i], sd);
      double noise = sd[2];
      double sigmaNoiseGeV = 1e-3 * noise * weight;
      make_rechit(i, dst_soa, src_soa, true, weight, 0., 0., sigmaNoiseGeV, sf);
    }
}
