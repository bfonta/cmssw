#ifndef CUDADataFormats_HGCal_HGCCLUESoA_h
#define CUDADataFormats_HGCal_HGCCLUESoA_h

#include <cstdint>

class HGCCLUEHitsSoA {
public:
  float *rho; //energy density of the calibrated rechit
  float *delta; //closest distance to a rechit with a higher density
  int32_t *nearestHigher; //index of the nearest rechit with a higher density
  int32_t *clusterIndex;  //cluster index the rechit belongs to
  bool *isSeed; // is the rechit a cluster seed?
  //Note: isSeed is of type int in the CPU version to to std::vector optimizations

  uint32_t nbytes;  //number of bytes of the SoA
  uint32_t nhits;   //number of hits stored in the SoA
  uint32_t pad;     //pad of memory block (used for warp alignment, slightly larger than 'nhits_')
};

class HGCCLUEClustersSoA {
public:
  float *energy; //energy of the cluster
  float *x; //x position of the cluster
  float *y; //y position of the cluster
  float *z; //z position of the cluster
  int32_t *clusterIndex;  //cluster index (matches the one from HGCCLUEHitsSoA)
    
  uint32_t nbytes;  //number of bytes of the SoA
  uint32_t nclusters;   //number of hits stored in the SoA
  uint32_t pad;     //pad of memory block (used for warp alignment, slightly larger than 'nclusters')
};

namespace memory {
  namespace npointers {
    //number of float pointers in the clue SoAs
    constexpr unsigned float_hgccluehits_soa = 2;
    constexpr unsigned float_hgcclueclusters_soa = 3;
    //number of int32 pointers in the clue SoAs
    constexpr unsigned int32_hgccluehits_soa = 2;
    constexpr unsigned int32_hgcclueclusters_soa = 1;
    //number of bool pointers in the clue SoAs
    constexpr unsigned bool_hgccluehits_soa = 1;
    constexpr unsigned bool_hgcclueclusters_soa = 0;
    //number of different pointer types in the clue SoAs
    constexpr unsigned ntypes_hgccluehits_soa = 3;
    constexpr unsigned ntypes_hgcclueclusters_soa = 2;
  } // namespace npointers
} // namespace memory

#endif  //CUDADataFormats_HGCal_HGCCLUESoA_h
