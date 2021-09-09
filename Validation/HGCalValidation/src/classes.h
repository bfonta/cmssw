#ifdef __CINT_
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclasses;
#pragma link C++ class ValidHit + ;
#pragma link C++ class vector < ValidHit> + ;
#pragma link C++ class ValidCLUECluster + ;
#pragma link C++ class vector < ValidCLUECluster> + ;
#endif /* __CINT__ */

#include "Validation/HGCalValidation/interface/ValidHit.h"
#include "Validation/HGCalValidation/interface/ValidCLUECluster.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"

ValidHit vh;
std::vector<ValidHit> vvh;
edm::Wrapper<std::vector<ValidHit> > wvvh;

ValidCLUECluster vcc;
std::vector<ValidCLUECluster> vvcc;
edm::Wrapper<std::vector<ValidCLUECluster> > wvvcc;
