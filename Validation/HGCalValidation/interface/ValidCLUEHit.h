#ifndef Validation_HGCalValidation_ValidCLUEHit_h
#define Validation_HGCalValidation_ValidCLUEHit_h

#include "TObject.h"
#include <iostream>
#include <vector>

class ValidCLUEHit {
public:
  ValidCLUEHit() : rho_(0.f), delta_(0.f), x_(0.f), y_(0.f), nearestHigher_(0), clusterIndex_(0), layer_(0), id_(0), isSeed_(false) {}
  ValidCLUEHit(float rho, float delta, float x, float y, int32_t nearestHigher, int32_t clusterIndex, int32_t layer, uint32_t id, bool isSeed)
    : rho_(rho), delta_(delta), x_(x), y_(y), nearestHigher_(nearestHigher), clusterIndex_(clusterIndex), layer_(layer), id_(id), isSeed_(isSeed) {}
  ValidCLUEHit(const ValidCLUEHit &other) {
    rho_ = other.rho_;
    delta_ = other.delta_;
    x_ = other.x_;
    y_ = other.y_;
    nearestHigher_ = other.nearestHigher_;
    clusterIndex_ = other.clusterIndex_;
    id_ = other.id_;
    layer_ = other.layer_;
    isSeed_ = other.isSeed_;
  }

  virtual ~ValidCLUEHit() {}

  float rho() { return rho_; }
  float delta() { return delta_; }
  float x() { return x_; }
  float y() { return y_; }
  int32_t nearestHigher() { return nearestHigher_; }
  int32_t clusterIndex() { return clusterIndex_; }
  uint32_t layer() { return layer_; }
  uint32_t id() { return id_; }
  bool isSeed() { return isSeed_; }

  float rho_; //energy density of the calibrated rechit
  float delta_; //closest distance to a rechit with a higher density
  float x_; //x hit position
  float y_; //y hit position
  int32_t nearestHigher_; //index of the nearest rechit with a higher density
  int32_t clusterIndex_;  //cluster index the rechit belongs to
  int32_t layer_; ////layer the rechit belongs to
  uint32_t id_; //rechit detId
  bool isSeed_; // is the rechit a cluster seed?
  ClassDef(ValidCLUEHit, 1)
};

typedef std::vector<ValidCLUEHit> ValidCLUEHitCollection;

#endif  //Validation_HGCalValidation_ValidCLUEHit_h
