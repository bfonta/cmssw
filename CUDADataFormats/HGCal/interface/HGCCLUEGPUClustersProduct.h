#ifndef CUDADAtaFormats_HGCal_HGCCLUEGPUClustersProduct_H
#define CUDADAtaFormats_HGCal_HGCCLUEGPUClustersProduct_H

#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCCLUEGPUClustersProduct {
public:
  HGCCLUEGPUClustersProduct() = default;
  explicit HGCCLUEGPUClustersProduct(uint32_t nclusters, const cudaStream_t &stream) : nclusters_(nclusters) {
    size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);
    pad_ = ((nclusters - 1) / 32 + 1) * 32; //align to warp boundary (assumption: warpSize = 32)
    mMemCLUEClustersDev = cms::cuda::make_device_unique<std::byte[]>(pad_ * size_tot_, stream);
  }
  ~HGCCLUEGPUClustersProduct() = default;

  HGCCLUEGPUClustersProduct(const HGCCLUEGPUClustersProduct &) = delete;
  HGCCLUEGPUClustersProduct &operator=(const HGCCLUEGPUClustersProduct &) = delete;
  HGCCLUEGPUClustersProduct(HGCCLUEGPUClustersProduct &&) = default;
  HGCCLUEGPUClustersProduct &operator=(HGCCLUEGPUClustersProduct &&) = default;

  HGCCLUEClustersSoA get() {
    HGCCLUEClustersSoA soa;
    soa.energy = reinterpret_cast<float *>(mMemCLUEClustersDev.get());
    soa.x = soa.energy + pad_;
    soa.y = soa.x + pad_;
    soa.z = soa.y + pad_;
    soa.clusterIndex = reinterpret_cast<int32_t *>(soa.z + pad_);
    return soa;
  }

  ConstHGCCLUEClustersSoA get() const {
    ConstHGCCLUEClustersSoA soa;
    soa.energy = reinterpret_cast<float const*>(mMemCLUEClustersDev.get());
    soa.x = soa.energy + pad_;
    soa.y = soa.x + pad_;
    soa.z = soa.y + pad_;
    soa.clusterIndex = reinterpret_cast<int32_t const*>(soa.z + pad_);
    return soa;
  }

  //number of clusters stored in the SoA
  uint32_t nClusters() const { return nclusters_; }
  //pad of memory block (used for warp alignment, slighlty larger than 'nclusters_')
  uint32_t pad() const { return pad_; }
  //number of bytes of the SoA
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::device::unique_ptr<std::byte[]> mMemCLUEClustersDev;
  static constexpr std::array<uint32_t, memory::npointers::ntypes_hgcclueclusters_soa> sizes_ = {
      {memory::npointers::float_hgcclueclusters_soa * sizeof(float),
       memory::npointers::int32_hgcclueclusters_soa * sizeof(uint32_t)}};
  uint32_t pad_;
  uint32_t nclusters_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCCLUEGPUClustersProduct_H
