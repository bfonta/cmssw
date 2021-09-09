#ifndef _HeterogeneousHGCalCLUEValidator_h
#define _HeterogeneousHGCalCLUEValidator_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Validation/HGCalValidation/interface/ValidCLUECluster.h"

#include "TTree.h"
#include "TH1F.h"

#include <iostream>
#include <string>

class HeterogeneousHGCalCLUEValidator : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HeterogeneousHGCalCLUEValidator(const edm::ParameterSet&);
  ~HeterogeneousHGCalCLUEValidator() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  static const unsigned nsubdetectors = 1;      //EM e/ou HAD
  static const unsigned ncomputingdevices = 2;  //cpu, gpu
  //cpu amd gpu tokens and handles for the 3 subdetectors, cpu and gpu
  std::array<std::array<edm::EDGetTokenT<reco::BasicClusterCollection>, ncomputingdevices>, nsubdetectors> tokens_;

  hgcal::RecHitTools recHitTools_;

  std::array<TTree*, nsubdetectors> trees_;
  std::array<std::string, nsubdetectors> treenames_;
  std::array<ValidCLUEClusterCollection, nsubdetectors> cpuValidCLUEHits, gpuValidCLUEHits, diffsValidCLUEHits;
};

#endif //_HeterogeneousHGCalCLUEValidator_h
