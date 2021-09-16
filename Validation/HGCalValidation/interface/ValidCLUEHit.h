#ifndef Validation_HGCalValidation_ValidCLUEHit_h
#define Validation_HGCalValidation_ValidCLUEHit_h

#include "TObject.h"
#include <iostream>
#include <vector>

class ValidCLUEHit {
public:
  ValidCLUEHit() : rho_(0), delta_(0), nearestHigher_(0), clusterIndex_(0), id_(0), isSeed_(false) {}
  ValidCLUEHit(float rho, float delta, int32_t nearestHigher, int32_t clusterIndex, uint32_t id, bool isSeed)
    : rho_(rho), delta_(delta), nearestHigher_(nearestHigher), clusterIndex_(clusterIndex), id_(id), isSeed_(isSeed) {}
  ValidCLUEHit(const ValidCLUEHit &other) {
    rho_ = other.rho_;
    delta_ = other.delta_;
    nearestHigher_ = other.nearestHigher_;
    clusterIndex_ = other.clusterIndex_;
    id_ = other.id_;
    isSeed_ = other.isSeed_;
  }

  virtual ~ValidCLUEHit() {}

  double rho() { return rho_; }
  double delta() { return delta_; }
  int32_t nearestHigher() { return nearestHigher_; }
  int32_t clusterIndex() { return clusterIndex_; }
  uint32_t id() { return id_; }
  bool isSeed() { return isSeed_; }

  float rho_; //energy density of the calibrated rechit
  float delta_; //closest distance to a rechit with a higher density
  int32_t nearestHigher_; //index of the nearest rechit with a higher density
  int32_t clusterIndex_;  //cluster index the rechit belongs to
  uint32_t id_; //rechit detId
  bool isSeed_; // is the rechit a cluster seed?
  ClassDef(ValidCLUEHit, 1)
};

typedef std::vector<ValidCLUEHit> ValidCLUEHitCollection;

#endif  //Validation_HGCalValidation_ValidCLUEHit_h
