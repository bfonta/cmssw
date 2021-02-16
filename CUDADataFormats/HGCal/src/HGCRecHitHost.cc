#include "CUDADataFormats/HGCal/interface/HGCRecHitHost.h"

std::unique_ptr<HGCRecHitSoA> layoutHGCRecHitHost(uint32_t nhits, const cudaStream_t& stream) {
  constexpr std::array<int,memory::npointers::ntypes_hgcrechits_soa> sizes = {{ memory::npointers::float_hgcrechits_soa * sizeof(float),
										memory::npointers::uint32_hgcrechits_soa * sizeof(uint32_t),
										memory::npointers::uint8_hgcrechits_soa * sizeof(uint8_t) }};

  uint32_t size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
  uint32_t pad = ((nhits - 1) / 32 + 1) * 32;  //align to warp boundary (assumption: warpSize = 32)
  cms::cuda::host::unique_ptr<std::byte[]> ptr_ = cms::cuda::make_host_unique<std::byte[]>(pad * size_tot, stream);

  std::unique_ptr<HGCRecHitSoA> soa = std::make_unique<HGCRecHitSoA>();
  soa->energy_ = reinterpret_cast<float*>(ptr_.get());
  soa->time_ = soa->energy_ + pad;
  soa->timeError_ = soa->time_ + pad;
  soa->id_ = reinterpret_cast<uint32_t*>(soa->timeError_ + pad);
  soa->flagBits_ = soa->id_ + pad;
  soa->son_ = reinterpret_cast<uint8_t*>(soa->flagBits_ + pad);

  soa->nbytes_ = size_tot;
  soa->nhits_ = nhits;
  soa->pad_ = pad;

  //return std::make_pair<cms::cuda::make_host_unique<std::byte[], std::unique_ptr<HGCRecHitSoA>>(ptr_, soaPtr);
  return soa;
}
