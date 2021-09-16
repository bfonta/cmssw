#ifndef RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgoGPUEMKernelImpl_cuh
#define RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgoGPUEMKernelImpl_cuh

#include <cuda_runtime.h>
#include <cuda.h>

#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUESoA.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUBase.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEM.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

__device__
bool is_energy_valid(float en);

__device__
float distance2(int id1, int id2, const clue_gpu::HGCCLUEInputSoAEM& in);

__device__
void get_total_cluster_weight(float& totalWeight, float& maxWeight, int& maxWeightId,
			      int seedId,
			      const clue_gpu::HGCCLUEInputSoAEM& in,
			      const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers);

__device__
void recursive_calculation(float& x, float& y, float& partialWeight,
			   float totalWeight,
			   float dc2,
			   int seedId, int maxWeightId, 
			   const clue_gpu::HGCCLUEInputSoAEM& in,
			   const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers);

__global__
void kernel_fill_input_soa(ConstHGCRecHitSoA hits,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds,
			   float ecut);

__global__
void kernel_compute_histogram( HeterogeneousHGCalLayerTiles *hist,
			       clue_gpu::HGCCLUEInputSoAEM in,
			       int numberOfPoints
			       );

__global__
void kernel_calculate_density( HeterogeneousHGCalLayerTiles *hist, 
			       clue_gpu::HGCCLUEInputSoAEM in,
			       HGCCLUEHitsSoA out,
			       float dc,
			       int numberOfPoints
			       );

__global__
void kernel_calculate_distanceToHigher(HeterogeneousHGCalLayerTiles* hist, 
				       clue_gpu::HGCCLUEInputSoAEM in,
				       HGCCLUEHitsSoA out,
				       float outlierDeltaFactor,
				       float dc,
				       int numberOfPoints
				       );

__global__
void kernel_find_clusters( cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* d_seeds,
			   cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* d_followers,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   HGCCLUEHitsSoA out,
			   float outlierDeltaFactor, float dc, float kappa,
			   int numberOfPoints
			   );

__global__
void kernel_assign_clusters( const cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* d_seeds, 
			     const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* d_followers,
			     HGCCLUEHitsSoA out);

__global__
void kernel_get_clusters(float dc2,
			 const cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* dSeeds,
			 const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers,
			 clue_gpu::HGCCLUEInputSoAEM hitsIn,
			 HGCCLUEHitsSoA hitsOut,
			 HGCCLUEClustersSoA clustersSoA,
			 unsigned nClustersPerLayer);

#endif //RecoLocalCalo_HGCalRecProducers_HGCalCLUEAlgoGPUEMKernelImpl_cuh
