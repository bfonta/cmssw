#include "UserCode/RecHitsValidator/plugins/HeterogeneousHGCalRecHitsValidator.h"

#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

HeterogeneousHGCalRecHitsValidator::HeterogeneousHGCalRecHitsValidator( const edm::ParameterSet &ps ) : 
  tokens_( {{ {{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsHSiToken")),
		consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsHSiToken"))}},
	      {{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsHSiToken")),
		consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsHSiToken"))}},
	      {{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("cpuRecHitsHSiToken")),
		consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("gpuRecHitsHSiToken"))}} }} ),
  treename_("tree")
{
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>(treename_.c_str(), treename_.c_str());
  tree_->Branch( "cpuDetid", "std::vector<unsigned int>", &cpuValidRecHits.detid );
  tree_->Branch( "gpuDetid", "std::vector<unsigned int>", &gpuValidRecHits.detid );
  //zhist.push_back(fs->make<TH1F>(("z"+treename_).c_str(), ("z"+treename_).c_str(), 2000, -10, 650));
}

HeterogeneousHGCalRecHitsValidator::~HeterogeneousHGCalRecHitsValidator()
{
}

void HeterogeneousHGCalRecHitsValidator::endJob()
{
}

void HeterogeneousHGCalRecHitsValidator::set_geometry_(const edm::EventSetup& setup, const unsigned int& detidx)
{
  edm::ESHandle<HGCalGeometry> handle;
  setup.get<IdealGeometryRecord>().get(handles_str_[detidx], handle);
}

void HeterogeneousHGCalRecHitsValidator::analyze(const edm::Event &event, const edm::EventSetup &setup)
{
  recHitTools_.getEventSetup(setup);
    
  //future subdetector loop
  for(size_t idet=0; idet<1; ++idet) {
    set_geometry_(setup, 1/*idet*/);

    //get hits produced with the CPU
    event.getByToken(tokens_[idet][0], handles_[idet][0]);
    const auto &cpuhits = *handles_[idet][0];

    //get hits produced with the GPU
    event.getByToken(tokens_[idet][1], handles_[idet][1]);
    const auto &gpuhits = *handles_[idet][1];

    size_t nhits = cpuhits.size();
    assert( nhits == gpuhits.size() );
    for(unsigned int i=0; i<nhits; i++) {
      const HGCRecHit &cpuHit = cpuhits[i];
      const HGCRecHit &gpuHit = gpuhits[i];

      /*
      const HGCalDetId cpuDetid = cpuHit.detid();
      const HGCalDetId gpuDetid = gpuHit.detid();
      */
      const float cpuSoN = cpuHit.signalOverSigmaNoise();
      const float gpuSoN = gpuHit.signalOverSigmaNoise();
      /*
      cpuValidRecHits.detid.push_back( cpuDetid );
      gpuValidRecHits.detid.push_back( gpuDetid );
      */
      std::cout << gpuSoN << ", " << cpuSoN << std::endl;
    }
  }    
  tree_->Fill();
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeterogeneousHGCalRecHitsValidator);
