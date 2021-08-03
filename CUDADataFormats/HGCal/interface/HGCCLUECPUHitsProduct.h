#ifndef CUDADAtaFormats_HGCal_HGCCLUECPUHitsProduct_H
#define CUDADAtaFormats_HGCal_HGCCLUECPUHitsProduct_H

#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCCLUECPUHitsProduct {
public:
  HGCCLUECPUHitsProduct() = default;
  explicit HGCCLUECPUHitsProduct(uint32_t nhits, const cudaStream_t &stream) : nhits_(nhits) {
    size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);
    pad_ = ((nhits - 1) / 32 + 1) * 32; //align to warp boundary (assumption: warpSize = 32)
    mMemCLUEHost = cms::cuda::make_host_unique<std::byte[]>(pad_ * size_tot_, stream);
  }
  ~HGCCLUECPUHitsProduct() = default;

  HGCCLUECPUHitsProduct(const HGCCLUECPUHitsProduct &) = delete;
  HGCCLUECPUHitsProduct &operator=(const HGCCLUECPUHitsProduct &) = delete;
  HGCCLUECPUHitsProduct(HGCCLUECPUHitsProduct &&) = default;
  HGCCLUECPUHitsProduct &operator=(HGCCLUECPUHitsProduct &&) = default;

  HGCCLUEHitsSoA get() {
    HGCCLUEHitsSoA soa;
    soa.rho = reinterpret_cast<float *>(mMemCLUEHost.get());
    soa.delta = soa.rho + pad_;
    soa.nearestHigher = reinterpret_cast<int32_t *>(soa.delta + pad_);
    soa.clusterIndex = soa.nearestHigher + pad_;
    soa.isSeed = reinterpret_cast<bool *>(soa.clusterIndex + pad_);
    soa.nbytes = size_tot_;
    soa.nhits = nhits_;
    soa.pad = pad_;
    return soa;
  }

  ConstHGCCLUEHitsSoA get() const {
    ConstHGCCLUEHitsSoA soa;
    soa.rho = reinterpret_cast<float const*>(mMemCLUEHost.get());
    soa.delta = soa.rho + pad_;
    soa.nearestHigher = reinterpret_cast<int32_t const*>(soa.delta + pad_);
    soa.clusterIndex = soa.nearestHigher + pad_;
    soa.isSeed = reinterpret_cast<bool const*>(soa.clusterIndex + pad_);
    return soa;
  }

  //number of hits stored in the SoA
  uint32_t nHits() const { return nhits_; }
  //pad of memory block (used for warp alignment, slighlty larger than 'nhits_')
  uint32_t pad() const { return pad_; }
  //number of bytes of the SoA
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::host::unique_ptr<std::byte[]> mMemCLUEHost;
  static constexpr std::array<uint32_t, memory::npointers::ntypes_hgccluehits_soa> sizes_ = {
      {memory::npointers::float_hgccluehits_soa * sizeof(float),
       memory::npointers::int32_hgccluehits_soa * sizeof(uint32_t),
       memory::npointers::bool_hgccluehits_soa * sizeof(bool)}};
  uint32_t pad_;
  uint32_t nhits_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCCLUECPUHitsProduct_H
