#ifndef CUDADataFormats_HGCal_ConstHGCCLUESoA_h
#define CUDADataFormats_HGCal_ConstHGCCLUESoA_h

#include <cstdint>

//const version of the HGCCLUEHitsSoA class (data in the event should be immutable)
class ConstHGCCLUEHitsSoA {
public:
  float const *rho; //energy density of the calibrated rechit
  float const *delta; //closest distance to a rechit with a higher density
  int32_t const *nearestHigher; //index of the nearest rechit with a higher density
  int32_t const *clusterIndex;  //cluster index the rechit belongs to
  bool const *isSeed; // is the rechit a cluster seed?
  //Note: isSeed is of type int in the CPU version to to std::vector optimizations
};

//const version of the HGCCLUEClustersSoA class (data in the event should be immutable)
class ConstHGCCLUEClustersSoA {
public:
  float const *energy; //energy of the cluster
  float const *x; //x position of the cluster
  float const *y; //y position of the cluster
  float const *z; //z position of the cluster
  int32_t const *clusterIndex;  //cluster index (matches the one from HGCCLUEHitsSoA)
};

#endif  //CUDADataFormats_HGCal_ConstHGCCLUESoA_h
