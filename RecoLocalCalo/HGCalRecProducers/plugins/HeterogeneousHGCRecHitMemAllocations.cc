#include "HeterogeneousHGCRecHitMemAllocations.h"

namespace memory {
  namespace allocation {
    cms::cuda::device::unique_ptr<std::byte[]> uncalibRecHitDevice(const uint32_t& nhits,
                                                                   const uint32_t& pad,
                                                                   HGCUncalibratedRecHitSoA& soa,
                                                                   const cudaStream_t& stream) {
      std::vector<int> sizes = {npointers::float_hgcuncalibrechits_soa * sizeof(float),
                                npointers::uint32_hgcuncalibrechits_soa * sizeof(uint32_t)};
      int size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
      cms::cuda::device::unique_ptr<std::byte[]> mem =
          cms::cuda::make_device_unique<std::byte[]>(pad * size_tot, stream);

      soa.amplitude_ = reinterpret_cast<float*>(mem.get());
      soa.pedestal_ = soa.amplitude_ + pad;
      soa.jitter_ = soa.pedestal_ + pad;
      soa.chi2_ = soa.jitter_ + pad;
      soa.OOTamplitude_ = soa.chi2_ + pad;
      soa.OOTchi2_ = soa.OOTamplitude_ + pad;
      soa.flags_ = reinterpret_cast<uint32_t*>(soa.OOTchi2_ + pad);
      soa.aux_ = soa.flags_ + pad;
      soa.id_ = soa.aux_ + pad;

      soa.nbytes_ = size_tot;
      soa.nhits_ = nhits;
      soa.pad_ = pad;

      assert(sizes.begin() + npointers::ntypes_hgcuncalibrechits_soa == sizes.end());

      return mem;
    }

    cms::cuda::host::unique_ptr<std::byte[]> uncalibRecHitHost(const uint32_t& nhits,
							       const uint32_t& pad,
                                                               HGCUncalibratedRecHitSoA& soa,
                                                               const cudaStream_t& stream) {
      std::vector<int> sizes = {npointers::float_hgcuncalibrechits_soa * sizeof(float),
                                npointers::uint32_hgcuncalibrechits_soa * sizeof(uint32_t)};
      int size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
      cms::cuda::host::unique_ptr<std::byte[]> mem =
          cms::cuda::make_host_unique<std::byte[]>(pad * size_tot, stream);

      soa.amplitude_ = reinterpret_cast<float*>(mem.get());
      soa.pedestal_ = soa.amplitude_ + pad;
      soa.jitter_ = soa.pedestal_ + pad;
      soa.chi2_ = soa.jitter_ + pad;
      soa.OOTamplitude_ = soa.chi2_ + pad;
      soa.OOTchi2_ = soa.OOTamplitude_ + pad;
      soa.flags_ = reinterpret_cast<uint32_t*>(soa.OOTchi2_ + pad);
      soa.aux_ = soa.flags_ + pad;
      soa.id_ = soa.aux_ + pad;
      soa.aux_ = soa.flags_ + pad;
      soa.id_ = soa.aux_ + pad;

      soa.nbytes_ = size_tot;
      soa.nhits_ = nhits;
      soa.pad_ = pad;

      assert(sizes.begin() + npointers::ntypes_hgcuncalibrechits_soa == sizes.end());

      return mem;
    }

    cms::cuda::host::unique_ptr<std::byte[]> calibRecHitHost(const uint32_t& nhits,
                                                             const uint32_t& pad,
                                                             HGCRecHitSoA& soa,
                                                             const cudaStream_t& stream) {
      std::vector<int> sizes = {npointers::float_hgcrechits_soa * sizeof(float),
                                npointers::uint32_hgcrechits_soa * sizeof(uint32_t),
                                npointers::uint8_hgcrechits_soa * sizeof(uint8_t)};
      int size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
      cms::cuda::host::unique_ptr<std::byte[]> mem =
          cms::cuda::make_host_unique<std::byte[]>(pad * size_tot, stream);
      soa.energy_ = reinterpret_cast<float*>(mem.get());
      soa.time_ = soa.energy_ + pad;
      soa.timeError_ = soa.time_ + pad;
      soa.id_ = reinterpret_cast<uint32_t*>(soa.timeError_ + pad);
      soa.flagBits_ = soa.id_ + pad;
      soa.son_ = reinterpret_cast<uint8_t*>(soa.flagBits_ + pad);

      soa.nbytes_ = size_tot;
      soa.nhits_ = nhits;
      soa.pad_ = pad;

      assert(sizes.begin() + npointers::ntypes_hgcrechits_soa == sizes.end());

      return mem;
    }
  }  // namespace allocation
}  // namespace memory
