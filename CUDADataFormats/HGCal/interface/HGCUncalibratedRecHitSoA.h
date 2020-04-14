#ifndef CudaDataFormats_HGCal_HGCUncalibratedRecHitSoA_h
#define CudaDataFormats_HGCal_HGCUncalibratedRecHitSoA_h

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

#endif
