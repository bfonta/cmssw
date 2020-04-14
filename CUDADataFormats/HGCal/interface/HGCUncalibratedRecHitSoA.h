#ifndef CUDADATAFORMATS_HGCUNCALIBRATEDRECHITSOA_H
#define CUDADATAFORMATS_HGCUNCALIBRATEDRECHITSOA_H 1

class HGCUncalibratedRecHitSoA {
public:
  float *amplitude_;
  float *pedestal_;
  float *jitter_;
  float *chi2_;
  float *OOTamplitude_;
  float *OOTchi2_;
  uint32_t *flags_;
  uint32_t *aux_;
  uint32_t *id_;
  int nbytes_;
};

#endif //CUDADATAFORMATS_HGCUNCAIBRATEDRECHITSOA_H
