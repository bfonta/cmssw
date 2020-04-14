#ifndef CUDADATAFORMATS_HGCRECHITSOA_H
#define CUDADATAFORMATS_HGCRECHITSOA_H 1

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

#endif //CUDADATAFORMATS_HGCRECHITSOA_H
