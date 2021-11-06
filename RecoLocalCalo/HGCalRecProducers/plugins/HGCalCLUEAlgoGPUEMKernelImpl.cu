#include <cuda.h>
#include <cuda_runtime.h>

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cuh"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgoGPUEMKernelImpl.cuh"

#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

__inline__ __device__
bool is_energy_valid(float en) {
  return en >= 0.;
} // kernel

template <class T>
__inline__ __device__
int32_t shift_layer(T id) {
  T did(id);
  return did.zside() == -1 ? did.layer() : did.layer() + static_cast<int32_t>(NLAYERS)/2;
}

__global__
void kernel_fill_input_soa(ConstHGCRecHitSoA hits,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   const hgcal_conditions::HeterogeneousPositionsConditionsESProduct* conds,
			   float ecut)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < hits.nhits; i += blockDim.x * gridDim.x) {
    in.sigmaNoise[i] = hits.sigmaNoise[i];
    in.id[i] = hits.id[i];
    in.energy[i] = (hits.energy[i]<ecut*in.sigmaNoise[i]) ? -1.f : hits.energy[i];

    //logic in https://github.com/cms-sw/cmssw/blob/master/RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cu
    const unsigned shift = hash_function(hits.id[i], conds);

    in.x[i] = conds->posmap.x[shift];
    in.y[i] = conds->posmap.y[shift];
    
    if(shift < static_cast<unsigned>(conds->posmap.nCellsTot)) //silicon
      in.layer[i] = shift_layer<HeterogeneousHGCSiliconDetId>( hits.id[i] );
    else //scintillator
      in.layer[i] = shift_layer<HeterogeneousHGCScintillatorDetId>( hits.id[i] );
    
  }
} // kernel


__global__
void kernel_compute_histogram( HeterogeneousHGCalLayerTiles *hist,
			       clue_gpu::HGCCLUEInputSoAEM in,
			       int numberOfPoints
			       )
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid; i < numberOfPoints; i += blockDim.x * gridDim.x)
    {
      if( is_energy_valid(in.energy[i]) )
	// push index of points into tiles
	hist[in.layer[i]].fill(in.x[i], in.y[i], i);
    }
} // kernel

__global__
void kernel_calculate_density( HeterogeneousHGCalLayerTiles *hist, 
			       clue_gpu::HGCCLUEInputSoAEM in,
			       HGCCLUEHitsSoA out,
			       float dc,
			       int numberOfPoints
			       ) 
{ 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid; i < numberOfPoints; i += blockDim.x * gridDim.x)
    {
      double rhoi{0.};

      if( is_energy_valid(in.energy[i]) ) {
      
	int layeri = in.layer[i];
	float xi = in.x[i];
	float yi = in.y[i];
      
	// get search box 
	int4 search_box = hist[layeri].searchBox(xi-dc, xi+dc, yi-dc, yi+dc);

	// loop over bins in the search box
	for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
	  for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {

	    // get the id of this bin
	    int binId = hist[layeri].getGlobalBinByBin(xBin,yBin);
	    // get the size of this bin
	    int binSize  = hist[layeri][binId].size();

	    // interate inside this bin
	    for (int binIter = 0; binIter < binSize; binIter++) {
	      int j = hist[layeri][binId][binIter];

	      if( is_energy_valid(in.energy[j]) ) {
		// query N_{mDc}(i)
		float xj = in.x[j];
		float yj = in.y[j];
		float dist_ij = std::sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj));
		if(dist_ij <= dc) { 
		  // sum weights within N_{mDc}(i)
		  rhoi += (i == j ? 1.f : 0.5f) * in.energy[j];
		}
	      }
	  
	    } // end of interate inside this bin
	  }
	} // end of loop over bins in search box
      }

      out.rho[i] = (float)rhoi;
      out.id[i] = in.id[i];

      //for testing only
      out.x[i] = in.x[i];
      out.y[i] = in.y[i];
    }
} //kernel


__global__
void kernel_calculate_distanceToHigher(HeterogeneousHGCalLayerTiles* hist, 
				       clue_gpu::HGCCLUEInputSoAEM in,
				       HGCCLUEHitsSoA out,
				       float outlierDeltaFactor,
				       float dc,
				       int numberOfPoints
				       )
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float dm = outlierDeltaFactor * dc;

  for (int i = tid; i < numberOfPoints; i += blockDim.x * gridDim.x)
    {

      float deltai = std::numeric_limits<float>::max();
      int nearestHigheri = -1;

      if( is_energy_valid(in.energy[i]) ) {
      
	int layeri = in.layer[i];
	float xi = in.x[i];
	float yi = in.y[i];
	float rhoi = out.rho[i];

	// get search box 
	int4 search_box = hist[layeri].searchBox(xi-dm, xi+dm, yi-dm, yi+dm);

	// loop over all bins in the search box
	for(int xBin = search_box.x; xBin < search_box.y+1; ++xBin) {
	  for(int yBin = search_box.z; yBin < search_box.w+1; ++yBin) {
	    // get the id of this bin
	    int binId = hist[layeri].getGlobalBinByBin(xBin,yBin);
	    // get the size of this bin
	    int binSize  = hist[layeri][binId].size();

	    // interate inside this bin
	    for (int binIter = 0; binIter < binSize; binIter++) {
	      int j = hist[layeri][binId][binIter];

	      if( is_energy_valid(in.energy[j]) ) {
		// query N'_{dm}(i)
		float xj = in.x[j];
		float yj = in.y[j];
		float dist_ij = std::sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj));
		bool foundHigher = (out.rho[j] > rhoi);
		// in the rare case where rho is the same, use detid
		foundHigher = foundHigher || ( (out.rho[j] == rhoi) && (j>i));
		if(foundHigher && dist_ij <= dm) { // definition of N'_{dm}(i)
		  // find the nearest point within N'_{dm}(i)
		  if (dist_ij<deltai) {
		    // update deltai and nearestHigheri
		    deltai = dist_ij;
		    nearestHigheri = j;
		  }
		}
	      }
	    } // end of interate inside this bin
	  }
	} // end of loop over bins in search box

      }
    
      out.delta[i] = deltai;
      out.nearestHigher[i] = nearestHigheri;

    } //  for (unsigned i = tid; i < numberOfPoints; i += blockDim.x * gridDim.x)
  
} //kernel



__global__
void kernel_find_clusters( cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* dSeeds,
			   cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers,
			   clue_gpu::HGCCLUEInputSoAEM in,
			   HGCCLUEHitsSoA out,
			   float outlierDeltaFactor, float dc, float kappa,
			   int numberOfPoints
			   ) 
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;


  for (int i = tid; i < numberOfPoints; i += blockDim.x * gridDim.x)
    {
      // initialize clusterIndex
      out.clusterIndex[i] = -1;

      if (is_energy_valid(in.energy[i])) {
	assert(out.nearestHigher[i] > -1);
	  
	// determine seed or outlier
	float deltai = out.delta[i];
	float rhoi = out.rho[i];
	float rhoc = kappa * in.sigmaNoise[i];
	bool isSeed = (deltai > dc) && (rhoi >= rhoc);
	bool isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

	if (isSeed) {
	  // set isSeed as 1
	  out.isSeed[i] = 1;
	  dSeeds[0].push_back(i); // head of dSeeds
	} else {
	  if (!isOutlier) {
	    assert(out.nearestHigher[i] < numberOfPoints);
	    // register as follower of its nearest higher
	    dFollowers[out.nearestHigher[i]].push_back(i);  
	  }
	}
      }
    } // for
  
} //kernel


__global__
void kernel_assign_clusters( const cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* dSeeds, 
			     const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers,
			     HGCCLUEHitsSoA out)
{
  int idxCluster = blockIdx.x * blockDim.x + threadIdx.x;

  const auto& seeds = dSeeds[0];
  const auto nSeeds = seeds.size();

  for (int i = idxCluster; i < nSeeds; i += blockDim.x * gridDim.x)
    {
      int localStack[clue_gpu::localStackSizePerSeed] = {-1};
      int localStackSize = 0;
      
      // assign cluster to seed[i]
      int idxThisSeed = seeds[i];
      out.clusterIndex[idxThisSeed] = i;
      // push_back idxThisSeed to localStack
      localStack[localStackSize] = idxThisSeed;
      localStackSize++;
      // process all elements in localStack

      while (localStackSize>0) {
      	// get last element of localStack
      	int idxEndOflocalStack = localStack[localStackSize-1];
      	int temp_clusterIndex = out.clusterIndex[idxEndOflocalStack];
      	// pop_back last element of localStack
  	localStack[localStackSize-1] = -1;
      	localStackSize--;
	
	// loop over followers of last element of localStack
      	for( int j : dFollowers[idxEndOflocalStack]){
      	  // pass id to follower
      	  out.clusterIndex[j] = temp_clusterIndex;
      	  // push_back follower to localStack
      	  localStack[localStackSize] = j;
	  localStackSize++;
      	  assert(localStackSize <= clue_gpu::localStackSizePerSeed);
	  assert(idxEndOflocalStack <= clue_gpu::maxNFollowers);
	}
      
      }
      
    } // for
  
} // kernel

__inline__ __device__
float calculate_max_w(float en, float totalWeight) {
  assert(en<=totalWeight);
  //float Wi = std::max(thresholdW0_[thick] + std::log(en / totalWeight), 0.);
  return std::max(2.9 + std::log(en / totalWeight), 0.);
}
	   
__device__
void calculate_position_and_energy(float& clusterX, float& clusterY, float& clusterEnergy, float& maxLog,
				   float totalWeight,
				   float dc2,
				   int seedId, int maxWeightId, 
				   const clue_gpu::HGCCLUEInputSoAEM& in,
				   const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers)
{
  for( int f : dFollowers[seedId])
    {
      float en = in.energy[f];
      clusterEnergy += en;
      
      if(distance2(f, maxWeightId, in) <= dc2)
	{
	  float Wi = calculate_max_w(en, totalWeight);
	  clusterX += in.x[f] * Wi;
	  clusterY += in.y[f] * Wi;
	  maxLog += Wi;
		
	  //if( dFollowers[f].size()!=0 ) //hit has at least one follower
	  calculate_position_and_energy(clusterX, clusterY, clusterEnergy, maxLog,
					totalWeight, dc2, f, maxWeightId,
					in, dFollowers);
      
	}
    }
}

//for the scintillator modify according to
//https://github.com/b-fontana/cmssw/blob/clusters/RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgo.h#L216
__inline__ __device__
float distance2(int id1, int id2, const clue_gpu::HGCCLUEInputSoAEM& in)
{
  const float dx = in.x[id1] - in.x[id2];
  const float dy = in.y[id1] - in.y[id2];
  return dx*dx + dy*dy;
}

__device__
void get_total_cluster_weight(float& totalWeight, float& maxWeight, int& maxWeightId,
			      int seedId,
			      const clue_gpu::HGCCLUEInputSoAEM& in,
			      const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers)
{
  for( int f : dFollowers[seedId]) {
    totalWeight += in.energy[f];
    
    if(in.energy[f]>maxWeight) {
      maxWeight = in.energy[f];
      maxWeightId = f;
    } 
    
    //if( dFollowers[f].size()!=0 ) //hit has at least one follower
    get_total_cluster_weight(totalWeight, maxWeight, maxWeightId,
			     f, in, dFollowers);
  }    
}

__global__
void kernel_get_clusters(float dc2,
			 const cms::cuda::VecArray<int,clue_gpu::maxNSeeds>* dSeeds,
			 const cms::cuda::VecArray<int,clue_gpu::maxNFollowers>* dFollowers,
			 clue_gpu::HGCCLUEInputSoAEM hitsIn,
			 HGCCLUEHitsSoA hitsOut,
			 HGCCLUEClustersSoA clustersSoA)
{
  const auto& seeds = dSeeds[0];
  const unsigned nseeds = seeds.size();
  assert(nseeds <= clustersSoA.nclusters);
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < nseeds; i += blockDim.x * gridDim.x)
    {
      //each thread will take care of a single cluster
      int thisSeed = seeds[i];
      assert(thisSeed != -1);
      
      int maxEnergyIndex  = -1;
      float maxWeight     = 0.f;
      float totalWeight = hitsIn.energy[thisSeed];

      //modifies totalWeight and maxEnergyIndex
      get_total_cluster_weight(totalWeight, maxWeight, maxEnergyIndex,
  			       thisSeed,
  			       hitsIn,
  			       dFollowers);

      // if(i % 10000 == 0) {
      // 	printf("ThreadId (bef): %u ::: hitsOutId=%u, maxWeight=%f, maxEnIndex=%d\n", i, hitsOut.id[maxEnergyIndex], maxWeight, maxEnergyIndex);
      // }
      
      //the x, y and energy of the seed is not included in 'calculate_position_and_energy()'
      //its inclusion is not possible due to the recursive approach
      float clusterEnergy = hitsIn.energy[thisSeed];
      float Wi = calculate_max_w(clusterEnergy, totalWeight);
      float maxLog = Wi;
      float clusterX = hitsIn.x[thisSeed] * Wi;
      float clusterY = hitsIn.y[thisSeed] * Wi;
      //printf("Avant: nFollowers: %d, totalWeight: %f, maxLog: %f, hitsX: %f, hitsY: %f, Wi: %f, clusterEnergy: %f, thisSeed: %d, clusterX: %f, clusterY: %f\n", dFollowers[thisSeed].size(), totalWeight, maxLog, hitsIn.x[thisSeed], hitsIn.y[thisSeed], Wi, clusterEnergy, thisSeed, clusterX, clusterY);

      //modifies clusterX, clusterY, clusterEnergy and maxLog
      calculate_position_and_energy(clusterX, clusterY, clusterEnergy, maxLog,
      				    totalWeight, 
      				    dc2,
      				    thisSeed,
      				    maxEnergyIndex,
      				    hitsIn,
      				    dFollowers );
      
      if( std::abs(maxLog) > 1e-10 ) {
	float inv = 1.f/maxLog;
	clusterX *= inv;
	clusterY *= inv;
      }
      else {
	printf("ZERO! (clusterEnergy=%f, totalWeight=%f, nFollowers=%u)\n",
	       clusterEnergy, totalWeight, dFollowers[thisSeed].size());
	clusterX = 0.f;
	clusterY = 0.f;
      }

      //printf("Apres: nFollowers: %d, totalWeight: %f, maxLog: %f, hitsX: %f, hitsY: %f, Wi: %f, clusterEnergy: %f, thisSeed %d, clusterX: %f, clusterY: %f\n", totalWeight, maxLog, hitsIn.x[thisSeed], hitsIn.y[thisSeed], Wi, clusterEnergy, thisSeed, clusterX, clusterY);
      //printf("---------------------\n");
      clustersSoA.energy[i] = clusterEnergy;
      clustersSoA.x[i]      = clusterX;
      clustersSoA.y[i]      = clusterY;
      clustersSoA.seedId[i] = (maxEnergyIndex==-1) ? hitsOut.id[thisSeed] : hitsOut.id[maxEnergyIndex];

      // if(i % 10000 == 0) {
      // 	printf("ThreadID (aft): %u ::: clusterSeedId=%u, hitsOutId=%u, hitOutsId=%u, thisSeed=%d, maxWeight=%f, maxEnIndex=%d, nFollowers=%u\n", i, clustersSoA.seedId[i], hitsOut.id[maxEnergyIndex], hitsOut.id[thisSeed], thisSeed, maxWeight, maxEnergyIndex, dFollowers[thisSeed].size());
      // 	printf("=========================\n");
      // }
      assert(hitsOut.id[maxEnergyIndex] == hitsOut.id[thisSeed]);
  }

} //kernel
