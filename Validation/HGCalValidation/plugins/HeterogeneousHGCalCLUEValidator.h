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
#include "Validation/HGCalValidation/interface/ValidCLUEHit.h"
#include "Validation/HGCalValidation/interface/ValidCLUECluster.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgo.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUECPUHitsProduct.h"

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HGCalReco/interface/CellsOnLayer.h"

#include "TTree.h"
#include "TH1F.h"

#include <iostream>
#include <string>

class HeterogeneousHGCalCLUEValidator : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  template <typename T, size_t SIZE>
  using Arr = std::array<T, SIZE>;

  using InClustersCPU = reco::BasicClusterCollection;
  using InClustersGPU = reco::BasicClusterCollection;
  using InHitsCPU = std::vector<CellsOnLayer>;
  using InHitsGPU = HGCCLUECPUHitsProduct;
  using OutCCol = ValidCLUEClusterCollection;
  using OutHCol = ValidCLUEHitCollection;
    
public:
  explicit HeterogeneousHGCalCLUEValidator(const edm::ParameterSet&);
  ~HeterogeneousHGCalCLUEValidator() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  static const unsigned nReg = 1;      //EM e/ou HAD
  static const unsigned nTechnologies = 2;      //CPU and GPU
  
  //cpu amd gpu tokens and handles for the 3 subdetectors, cpu and gpu
  Arr<edm::EDGetTokenT<InHitsCPU>, nReg> tokHitsCPU_;
  Arr<edm::EDGetTokenT<InHitsGPU>, nReg> tokHitsGPU_;
  Arr<edm::EDGetTokenT<InClustersCPU>, nReg> tokClustersCPU_;
  Arr<edm::EDGetTokenT<InClustersGPU>, nReg> tokClustersGPU_;


  TFile *outFile_;
  Arr<TTree*, nReg> treesH_, treesC_;
  Arr<TH1F*, nTechnologies> histosEn, histosX, histosY, histosZ;
  Arr<std::string, nReg> treenamesH_, treenamesC_;
  Arr<OutCCol, nReg> cpuValidClusters, gpuValidClusters, diffsValidClusters;
  Arr<OutHCol, nReg> cpuValidHits, gpuValidHits, diffsValidHits;
};

#endif //_HeterogeneousHGCalCLUEValidator_h
