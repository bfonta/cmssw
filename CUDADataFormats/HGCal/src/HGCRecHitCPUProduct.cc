#include "CUDADataFormats/HGCal/interface/HGCRecHitCPUProduct.h"

HGCRecHitCPUProduct::HGCRecHitCPUProduct(uint32_t nhits, const cudaStream_t& stream) : nhits_(nhits) {
  constexpr std::array<int,memory::npointers::ntypes_hgcrechits_soa> sizes = {{
				memory::npointers::float_hgcrechits_soa * sizeof(float),
				memory::npointers::uint32_hgcrechits_soa * sizeof(uint32_t),
				memory::npointers::uint8_hgcrechits_soa * sizeof(uint8_t) }};
  size_tot_ = std::accumulate(sizes.begin(), sizes.end(), 0);
  pad_ = ((nhits - 1) / 32 + 1) * 32;  //align to warp boundary (assumption: warpSize = 32)
  //mem_ = cms::cuda::make_host_unique<std::byte[]>(pad_ * size_tot_, stream);
  mem_ = std::make_unique<std::byte[]>(pad_ * size_tot_);

  defineSoAMemoryLayout_();
}

void HGCRecHitCPUProduct::defineSoAMemoryLayout_() {
  soa_.energy_    = reinterpret_cast<float*>(mem_.get());
  soa_.time_      = soa_.energy_ + pad_;
  soa_.timeError_ = soa_.time_ + pad_;
  soa_.id_        = reinterpret_cast<uint32_t*>(soa_.timeError_ + pad_);
  soa_.flagBits_  = soa_.id_ + pad_;
  soa_.son_       = reinterpret_cast<uint8_t*>(soa_.flagBits_ + pad_);

  soa_.nbytes_ = size_tot_;
  soa_.nhits_  = nhits_;
  soa_.pad_    = pad_;
}
