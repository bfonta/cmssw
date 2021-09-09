#ifndef Validation_HGCalValidation_ValidCLUECluster_h
#define Validation_HGCalValidation_ValidCLUECluster_h

#include "TObject.h"
#include <iostream>
#include <vector>

class ValidCLUECluster {
public:
  ValidCLUECluster() : energy_(0), x_(0), y_(0), z_(0) {}
  ValidCLUECluster(double energy, double x, double y, double z)
    : energy_(energy), x_(x), y_(y), z_(z) {}
  ValidCLUECluster(const ValidCLUECluster &other) {
    energy_ = other.energy_;
    x_ = other.x_;
    y_ = other.y_;
    z_ = other.z_;
  }

  virtual ~ValidCLUECluster() {}

  double energy() { return energy_; }
  double x() { return x_; }
  double y() { return y_; }
  double z() { return z_; }

  double energy_; //energy of the cluster
  double x_;      //x position of the cluster
  double y_;      //y position of the cluster
  double z_;      //z position of the cluster
  ClassDef(ValidCLUECluster, 1)
};

typedef std::vector<ValidCLUECluster> ValidCLUEClusterCollection;

#endif  //Validation_HGCalValidation_ValidCLUECluster_h
