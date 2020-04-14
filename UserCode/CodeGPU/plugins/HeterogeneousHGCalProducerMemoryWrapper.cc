#include "HeterogeneousHGCalProducerMemoryWrapper.h"
#include "Types.h"

namespace memory {
  namespace allocation {    
    namespace {
      //returns total number of bytes, number of 'double' elements and number of 'float' elements
      std::tuple<int, int, int, int> get_memory_sizes_(const std::vector<int>& fixed_sizes, const int& ndoubles, const int& nfloats, const int& nints)
      {
	assert( fixed_sizes.begin() + ndoubles + nfloats + nints == fixed_sizes.end() );
	const std::vector<int> sizes = {sizeof(double), sizeof(float), sizeof(int)};
	const std::vector<int> nelements = { std::accumulate( fixed_sizes.begin(), fixed_sizes.begin() + ndoubles, 0),
					     std::accumulate( fixed_sizes.begin() + ndoubles, fixed_sizes.begin() + ndoubles + nfloats, 0),
					     std::accumulate( fixed_sizes.begin() + ndoubles + nfloats, fixed_sizes.end(), 0) };
	int size_tot = std::inner_product(sizes.begin(), sizes.end(), nelements.begin(), 0);
	return std::make_tuple(size_tot, nelements[0], nelements[1], nelements[2]);
      }
    }

    //EE: allocates memory for constants on the device
    void device(KernelConstantData<HGCeeUncalibratedRecHitConstantData> *kcdata, cms::cuda::device::unique_ptr<double[]>& mem) {
      const std::vector<int> nelements = {kcdata->data_.s_hgcEE_fCPerMIP_, kcdata->data_.s_hgcEE_cce_, kcdata->data_.s_hgcEE_noise_fC_, kcdata->data_.s_rcorr_, kcdata->data_.s_weights_, kcdata->data_.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_device_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data_.hgcEE_fCPerMIP_ = mem.get();
      kcdata->data_.hgcEE_cce_      = kcdata->data_.hgcEE_fCPerMIP_ + nelements[0];
      kcdata->data_.hgcEE_noise_fC_ = kcdata->data_.hgcEE_cce_ + nelements[1];
      kcdata->data_.rcorr_          = kcdata->data_.hgcEE_noise_fC_ + nelements[2];
      kcdata->data_.weights_        = kcdata->data_.rcorr_ + nelements[3];
      kcdata->data_.waferTypeL_     = reinterpret_cast<int*>(kcdata->data_.weights_ + nelements[4]);
      kcdata->data_.nbytes_         = std::get<0>(memsizes);
      kcdata->data_.ndelem_         = std::get<1>(memsizes) + 2;
      kcdata->data_.nfelem_         = std::get<2>(memsizes) + 4;
      kcdata->data_.nielem_         = std::get<3>(memsizes) + 0;
      kcdata->data_.nuelem_         = 2;
    }

    //HEF: allocates memory for constants on the device
    void device(KernelConstantData<HGChefUncalibratedRecHitConstantData> *kcdata, cms::cuda::device::unique_ptr<double[]>& mem) {
      const std::vector<int> nelements = {kcdata->data_.s_hgcHEF_fCPerMIP_, kcdata->data_.s_hgcHEF_cce_, kcdata->data_.s_hgcHEF_noise_fC_, kcdata->data_.s_rcorr_, kcdata->data_.s_weights_, kcdata->data_.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_device_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data_.hgcHEF_fCPerMIP_ = mem.get();
      kcdata->data_.hgcHEF_cce_      = kcdata->data_.hgcHEF_fCPerMIP_ + nelements[0];
      kcdata->data_.hgcHEF_noise_fC_ = kcdata->data_.hgcHEF_cce_ + nelements[1];
      kcdata->data_.rcorr_           = kcdata->data_.hgcHEF_noise_fC_ + nelements[2];
      kcdata->data_.weights_         = kcdata->data_.rcorr_ + nelements[3];
      kcdata->data_.waferTypeL_      = reinterpret_cast<int*>(kcdata->data_.weights_ + nelements[4]);
      kcdata->data_.nbytes_          = std::get<0>(memsizes);
      kcdata->data_.ndelem_          = std::get<1>(memsizes) + 2;
      kcdata->data_.nfelem_          = std::get<2>(memsizes) + 4;
      kcdata->data_.nielem_          = std::get<3>(memsizes) + 0;
      kcdata->data_.nuelem_          = 3;
    }

    //HEB: allocates memory for constants on the device
    void device(KernelConstantData<HGChebUncalibratedRecHitConstantData> *kcdata, cms::cuda::device::unique_ptr<double[]>& mem) {
      const std::vector<int> nelements = {kcdata->data_.s_weights_};
      auto memsizes = get_memory_sizes_(nelements, 1, 0, 0);
      mem = cms::cuda::make_device_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data_.weights_  = mem.get();
      kcdata->data_.nbytes_   = std::get<0>(memsizes);
      kcdata->data_.ndelem_   = std::get<1>(memsizes) + 3;
      kcdata->data_.nfelem_   = std::get<2>(memsizes) + 0;
      kcdata->data_.nielem_   = std::get<3>(memsizes) + 0;
      kcdata->data_.nuelem_   = 3;
    }

    //allocates memory for UncalibratedRecHits SoAs and RecHits SoAs on the device
    void device(const int& nhits, HGCUncalibratedRecHitSoA* soa1, HGCUncalibratedRecHitSoA* soa2, HGCRecHitSoA* soa3, cms::cuda::device::unique_ptr<float[]>& mem)
    {
      std::vector<int> sizes = {6*sizeof(float), 3*sizeof(uint32_t),                     //soa1
				6*sizeof(float), 3*sizeof(uint32_t),                     //soa2
				3*sizeof(float), 2*sizeof(uint32_t), 1*sizeof(uint8_t)}; //soa3
      int size_tot = std::accumulate( sizes.begin(), sizes.end(), 0);
      mem = cms::cuda::make_device_unique<float[]>(nhits * size_tot, 0);

      soa1->amplitude_     = mem.get();
      soa1->pedestal_      = soa1->amplitude_    + nhits;
      soa1->jitter_        = soa1->pedestal_     + nhits;
      soa1->chi2_          = soa1->jitter_       + nhits;
      soa1->OOTamplitude_  = soa1->chi2_         + nhits;
      soa1->OOTchi2_       = soa1->OOTamplitude_ + nhits;
      soa1->flags_         = reinterpret_cast<uint32_t*>(soa1->OOTchi2_ + nhits);
      soa1->aux_           = soa1->flags_        + nhits;
      soa1->id_            = soa1->aux_          + nhits;

      soa2->amplitude_     = reinterpret_cast<float*>(soa1->id_ + nhits);
      soa2->pedestal_      = soa2->amplitude_    + nhits;
      soa2->jitter_        = soa2->pedestal_     + nhits;
      soa2->chi2_          = soa2->jitter_       + nhits;
      soa2->OOTamplitude_  = soa2->chi2_         + nhits;
      soa2->OOTchi2_       = soa2->OOTamplitude_ + nhits;
      soa2->flags_         = reinterpret_cast<uint32_t*>(soa2->OOTchi2_ + nhits);
      soa2->aux_           = soa2->flags_        + nhits;
      soa2->id_            = soa2->aux_          + nhits;
  
      soa3->energy_        = reinterpret_cast<float*>(soa2->id_ + nhits);
      soa3->time_          = soa3->energy_       + nhits;
      soa3->timeError_     = soa3->time_         + nhits;
      soa3->id_            = reinterpret_cast<uint32_t*>(soa3->timeError_ + nhits);
      soa3->flagBits_      = soa3->id_           + nhits;
      soa3->son_           = reinterpret_cast<uint8_t*>(soa3->flagBits_ + nhits);

      soa1->nbytes_ = std::accumulate(sizes.begin(), sizes.begin()+2, 0);
      soa2->nbytes_ = std::accumulate(sizes.begin()+2, sizes.begin()+4, 0);
      soa3->nbytes_ = std::accumulate(sizes.begin()+4, sizes.end(), 0);
    }

    //EE: allocates memory for constants on the host
    void host(KernelConstantData<HGCeeUncalibratedRecHitConstantData>* kcdata, cms::cuda::host::noncached::unique_ptr<double[]>& mem)
    {
      const std::vector<int> nelements = {kcdata->data_.s_hgcEE_fCPerMIP_, kcdata->data_.s_hgcEE_cce_, kcdata->data_.s_hgcEE_noise_fC_, kcdata->data_.s_rcorr_, kcdata->data_.s_weights_, kcdata->data_.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_host_noncached_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data_.hgcEE_fCPerMIP_ = mem.get();
      kcdata->data_.hgcEE_cce_      = kcdata->data_.hgcEE_fCPerMIP_ + nelements[0];
      kcdata->data_.hgcEE_noise_fC_ = kcdata->data_.hgcEE_cce_ + nelements[1];
      kcdata->data_.rcorr_          = kcdata->data_.hgcEE_noise_fC_ + nelements[2];
      kcdata->data_.weights_        = kcdata->data_.rcorr_ + nelements[3];
      kcdata->data_.waferTypeL_     = reinterpret_cast<int*>(kcdata->data_.weights_ + nelements[4]);
      kcdata->data_.nbytes_         = std::get<0>(memsizes);
      kcdata->data_.ndelem_         = std::get<1>(memsizes) + 2;
      kcdata->data_.nfelem_         = std::get<2>(memsizes) + 0;
      kcdata->data_.nielem_         = std::get<3>(memsizes) + 0;
      kcdata->data_.nuelem_         = 2;
    }

    //HEF: allocates memory for constants on the host
    void host(KernelConstantData<HGChefUncalibratedRecHitConstantData>* kcdata, cms::cuda::host::noncached::unique_ptr<double[]>& mem)
    {
      const std::vector<int> nelements = {kcdata->data_.s_hgcHEF_fCPerMIP_, kcdata->data_.s_hgcHEF_cce_, kcdata->data_.s_hgcHEF_noise_fC_, kcdata->data_.s_rcorr_, kcdata->data_.s_weights_, kcdata->data_.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_host_noncached_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data_.hgcHEF_fCPerMIP_ = mem.get();
      kcdata->data_.hgcHEF_cce_      = kcdata->data_.hgcHEF_fCPerMIP_ + nelements[0];
      kcdata->data_.hgcHEF_noise_fC_ = kcdata->data_.hgcHEF_cce_ + nelements[1];
      kcdata->data_.rcorr_           = kcdata->data_.hgcHEF_noise_fC_ + nelements[2];
      kcdata->data_.weights_         = kcdata->data_.rcorr_ + nelements[3];
      kcdata->data_.waferTypeL_      = reinterpret_cast<int*>(kcdata->data_.weights_ + nelements[4]);
      kcdata->data_.nbytes_          = std::get<0>(memsizes);
      kcdata->data_.ndelem_          = std::get<1>(memsizes) + 2;
      kcdata->data_.nfelem_          = std::get<2>(memsizes) + 0;
      kcdata->data_.nielem_          = std::get<3>(memsizes) + 0;
      kcdata->data_.nuelem_          = 3;
    }

    //HEB: allocates memory for constants on the host
    void host(KernelConstantData<HGChebUncalibratedRecHitConstantData>* kcdata, cms::cuda::host::noncached::unique_ptr<double[]>& mem)
    {
      const std::vector<int> nelements = {kcdata->data_.s_weights_};
      auto memsizes = get_memory_sizes_(nelements, 1, 0, 0);
      mem = cms::cuda::make_host_noncached_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data_.weights_ = mem.get();
      kcdata->data_.nbytes_  = std::get<0>(memsizes);
      kcdata->data_.ndelem_  = std::get<1>(memsizes) + 3;
      kcdata->data_.nfelem_  = std::get<2>(memsizes) + 0;
      kcdata->data_.nielem_  = std::get<3>(memsizes) + 0;
      kcdata->data_.nuelem_  = 3;
    }

    //allocates pinned (non cached) memory for UncalibratedRecHits SoAs on the host
    void host(const int& nhits, HGCUncalibratedRecHitSoA* soa, cms::cuda::host::noncached::unique_ptr<float[]>& mem)
    {
      std::vector<int> sizes = { 6*sizeof(float), 3*sizeof(uint32_t) };
      int size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
      mem = cms::cuda::make_host_noncached_unique<float[]>(nhits * size_tot, 0);

      soa->amplitude_     = mem.get();
      soa->pedestal_      = soa->amplitude_    + nhits;
      soa->jitter_        = soa->pedestal_     + nhits;
      soa->chi2_          = soa->jitter_       + nhits;
      soa->OOTamplitude_  = soa->chi2_         + nhits;
      soa->OOTchi2_       = soa->OOTamplitude_ + nhits;
      soa->flags_         = reinterpret_cast<uint32_t*>(soa->OOTchi2_ + nhits);
      soa->aux_           = soa->flags_        + nhits;
      soa->id_            = soa->aux_          + nhits;
      soa->nbytes_        = size_tot;
    }

    //allocates memory for RecHits SoAs on the host
    void host(const int& nhits, HGCRecHitSoA* soa, cms::cuda::host::unique_ptr<float[]>& mem)
    {
      std::vector<int> sizes = { 3*sizeof(float), 2*sizeof(uint32_t), sizeof(uint8_t) };
      int size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
      mem = cms::cuda::make_host_unique<float[]>(nhits * size_tot, 0);

      soa->energy_     = mem.get();
      soa->time_       = soa->energy_     + nhits;
      soa->timeError_  = soa->time_       + nhits;
      soa->id_         = reinterpret_cast<uint32_t*>(soa->timeError_ + nhits);
      soa->flagBits_   = soa->id_         + nhits;
      soa->son_        = reinterpret_cast<uint8_t*>(soa->flagBits_ + nhits);
      soa->nbytes_ = size_tot;
    }
  }
}
