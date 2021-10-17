#ifndef CUDADAtaFormats_HGCal_HGCCLUECPUClustersProduct_H
#define CUDADAtaFormats_HGCal_HGCCLUECPUClustersProduct_H

#include <numeric>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitSoA.h"

class HGCCLUECPUClustersProduct {
public:
  HGCCLUECPUClustersProduct() = default;
  explicit HGCCLUECPUClustersProduct(uint32_t nclusters, const cudaStream_t &stream) : nclusters_(nclusters) {
    size_tot_ = std::accumulate(sizes_.begin(), sizes_.end(), 0);
    pad_ = ((nclusters - 1) / 32 + 1) * 32; //align to warp boundary (assumption: warpSize = 32)
    mMemCLUEClustersHost = cms::cuda::make_host_unique<std::byte[]>(pad_ * size_tot_, stream);
  }
  ~HGCCLUECPUClustersProduct() = default;

  HGCCLUECPUClustersProduct(const HGCCLUECPUClustersProduct &) = delete;
  HGCCLUECPUClustersProduct &operator=(const HGCCLUECPUClustersProduct &) = delete;
  HGCCLUECPUClustersProduct(HGCCLUECPUClustersProduct &&) = default;
  HGCCLUECPUClustersProduct &operator=(HGCCLUECPUClustersProduct &&) = default;

  HGCCLUEClustersSoA get() {
    HGCCLUEClustersSoA soa;
    soa.energy = reinterpret_cast<float *>(mMemCLUEClustersHost.get());
    soa.x = soa.energy + pad_;
    soa.y = soa.x + pad_;
    soa.seedId = reinterpret_cast<uint32_t *>(soa.y + pad_);

    soa.nbytes = size_tot_;
    soa.nclusters = nclusters_;
    soa.pad = pad_;
    return soa;
  }

  ConstHGCCLUEClustersSoA get() const {
    ConstHGCCLUEClustersSoA soa;
    soa.energy = reinterpret_cast<float const*>(mMemCLUEClustersHost.get());
    soa.x = soa.energy + pad_;
    soa.y = soa.x + pad_;
    soa.seedId = reinterpret_cast<uint32_t const*>(soa.y + pad_);
    return soa;
  }

  //number of hits stored in the SoA
  uint32_t nClusters() const { return nclusters_; }
  //pad of memory block (used for warp alignment, slighlty larger than 'nclusters_')
  uint32_t pad() const { return pad_; }
  //number of bytes of the SoA
  uint32_t nBytes() const { return size_tot_; }

private:
  cms::cuda::host::unique_ptr<std::byte[]> mMemCLUEClustersHost;
  static constexpr std::array<uint32_t, memory::npointers::ntypes_hgcclueclusters_soa> sizes_ = {
      {memory::npointers::float_hgcclueclusters_soa * sizeof(float),
       memory::npointers::int32_hgcclueclusters_soa * sizeof(uint32_t)}};
  uint32_t pad_;
  uint32_t nclusters_;
  uint32_t size_tot_;
};

#endif  //CUDADAtaFormats_HGCal_HGCCLUECPUClustersProduct_H
