#ifndef CudaDataFormats_HGCal_HGCUncalibratedRecHitsToRecHitsConstants_h
#define CudaDataFormats_HGCal_HGCUncalibratedRecHitsToRecHitsConstants_h

#include <vector>

class HGCConstantVectorData {
 public:
  std::vector<double> fCPerMIP_;
  std::vector<double> cce_;
  std::vector<double> noise_fC_;
  std::vector<double> rcorr_;
  std::vector<double> weights_;
  std::vector<int> waferTypeL_;
};

class HGCeeUncalibratedRecHitConstantData {
 public:
  double hgcEE_keV2DIGI_; //energy to femto coloumb conversion: 1000 eV/3.62 (eV per e) / 6.24150934e3 (e per fC)
  double hgceeUncalib2GeV_; //sets the ADC; obtained by dividing 1e-6 by hgcEE_keV2DIGI_
  double *hgcEE_fCPerMIP_; //femto coloumb to MIP conversion; one value per sensor thickness
  double *hgcEE_cce_; //charge collection efficiency, one value per sensor thickness
  double *hgcEE_noise_fC_; //noise, one value per sensor thickness
  double *rcorr_; //thickness correction
  double *weights_; //energy weights to recover rechit energy deposited in the absorber
  int *waferTypeL_; //wafer longitudinal thickness classification (1 = 100um, 2 = 200um, 3=300um)
  float xmin_; //used for computing the time resolution error
  float xmax_; //used for computing the time resolution error
  float aterm_; //used for computing the time resolution error
  float cterm_; //used for computing the time resolution error
  int nbytes_; //number of bytes allocated by this class
  int ndelem_; //number of doubles pointed by this class
  int nfelem_; //number of floats pointed by this class
  int nielem_; //number of ints pointed by this class
  int s_hgcEE_fCPerMIP_; //number of elements pointed by hgcEE_fCPerMIP_
  int s_hgcEE_cce_; //number of elements pointed by hgcEE_cce_
  int s_hgcEE_noise_fC_; //number of elements pointed by hgcEE_noise_fC_
  int s_rcorr_; //number of elements pointed by rcorr_
  int s_weights_; //number of elements pointed by weights_
  int s_waferTypeL_; //number of elements pointed by waferTypeL_
};

class HGChefUncalibratedRecHitConstantData {
 public:
  double hgcHEF_keV2DIGI_; //energy to femto coloumb conversion: 1000 eV/3.62 (eV per e) / 6.24150934e3 (e per fC)
  double hgchefUncalib2GeV_; //sets the ADC; obtained by dividing 1e-6 by hgcHEF_keV2DIGI_
  double *hgcHEF_fCPerMIP_; //femto coloumb to MIP conversion; one value per sensor thickness
  double *hgcHEF_cce_; //charge collection efficiency, one value per sensor thickness
  double *hgcHEF_noise_fC_; //noise, one value per sensor thickness
  double *rcorr_;  //thickness correction
  double *weights_; //energy weights to recover rechit energy deposited in the absorber
  int *waferTypeL_; //wafer longitudinal thickness classification (1 = 100um, 2 = 200um, 3=300um)
  float xmin_; //used for computing the time resolution error
  float xmax_; //used for computing the time resolution error
  float aterm_; //used for computing the time resolution error
  float cterm_; //used for computing the time resolution error
  uint32_t fhOffset_; //layer offset
  int nbytes_; //number of bytes allocated by this class
  int ndelem_; //number of doubles allocated by this class
  int nfelem_; //number of floats allocated by this class
  int nuelem_; //number of unsigned ints allocated by this class
  int nielem_; //number of ints allocated by this class
  int s_hgcHEF_fCPerMIP_; //number of elements pointed by hgcEE_fCPerMIP_
  int s_hgcHEF_cce_; //number of elements pointed by hgcEE_cce_
  int s_hgcHEF_noise_fC_; //number of elements pointed by hgcEE_noise_fC_
  int s_rcorr_; //number of elements pointed by rcorr_
  int s_weights_; //number of elements pointed by weights_
  int s_waferTypeL_; //number of elements pointed by waferTypeL_
};

class HGChebUncalibratedRecHitConstantData {
 public:
  double hgcHEB_keV2DIGI_; //energy to femto coloumb conversion: 1000 eV/3.62 (eV per e) / 6.24150934e3 (e per fC)
  double hgchebUncalib2GeV_; //sets the ADC; obtained by dividing 1e-6 by hgcHEB_keV2DIGI_
  double hgcHEB_noise_MIP_; //noise
  double *weights_; //energy weights to recover rechit energy deposited in the absorber
  uint32_t bhOffset_; //layer offset
  int nbytes_; //number of bytes allocated by this class
  int ndelem_; //number of doubles allocated by this class
  int nfelem_; //number of floats allocated by this class
  int nuelem_; //number of unsigned ints allocated by this class
  int nielem_; //number of ints allocated by this class
  int s_weights_; //number of elements pointed by weights_
};

#endif
