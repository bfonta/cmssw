#ifndef CudaDataFormats_HGCal_HGCRecHitSoA_h
#define CudaDataFormats_HGCal_HGCRecHitSoA_h

class HGCRecHitSoA {
 public:
  float *energy_;
  float *time_;
  float *timeError_;
  uint32_t *id_;
  uint32_t *flagBits_;
  uint8_t *son_;
  int nbytes_;
};

#endif
