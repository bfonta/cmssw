#include "RecoLocalCalo/HGCalRecProducers/plugins/HeterogeneousHGCalHEFRecHitProducer.h"

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

HGCalMaskResolutionAna::HGCalMaskResolutionAna( const edm::ParameterSet &ps ) : 
  mc_( consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared")) ),
  recHitsTokens_( {consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("recHitsCEEToken")),
	consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("recHitsHSiToken")),
	consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("recHitsHScToken"))} ),
  genParticles_( consumes<std::vector<reco::GenParticle>>(edm::InputTag("genParticles"))),
  treename_("tree")
{  

  edm::Service<TFileService> fs;
  for(size_t idet = 0; idet<nSubDets_; ++idet) { 
    zhist.push_back(fs->make<TH1F>(("z"+treename_).c_str(), ("z"+treename_).c_str(), 2000, -10, 650));
    xhist.push_back(fs->make<TH1F>(("x"+treename_).c_str(), ("x"+treename_).c_str(), 1000, -40, 40));
    yhist.push_back(fs->make<TH1F>(("y"+treename_).c_str(), ("y"+treename_).c_str(), 1000, -40, 40));
    zsidehist.push_back(fs->make<TH1F>(("zside"+treename_).c_str(), ("zside"+treename_).c_str(), 4, -1.1, 1.1));
    ehist.push_back(fs->make<TH1F>(("e"+treename_).c_str(), ("e"+treename_).c_str(), 2000, -10, 650));
    layerhist.push_back(fs->make<TH1F>(("layer"+treename_).c_str(), ("layer"+treename_).c_str(), 102, 0, 52));
    offsetlayerhist.push_back(fs->make<TH1F>(("offsetlayer"+treename_).c_str(), ("offsetlayer"+treename_).c_str(), 102, 0, 52));
  }

  tree_ = fs->make<TTree>((treenames_[0]+"_"+treenames_[1]+"_"+treenames_[2]).c_str(), 
			  (treenames_[0]+"_"+treenames_[1]+"_"+treenames_[2]).c_str());
  tree_->Branch("Hits", "std::vector<SlimmedHit>", &slimmedHits_);
  tree_->Branch("ROIs", "std::vector<SlimmedROI>", &slimmedROIs_);
  tree_->Branch("GenVertex", "TLorentzVector", &genVertex_);
}

HGCalMaskResolutionAna::~HGCalMaskResolutionAna()
{
}

void HGCalMaskResolutionAna::endJob()
{
}

void HGCalMaskResolutionAna::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  recHitTools_.getEventSetup(iSetup);

  std::vector< edm::Handle<HGCRecHitCollection> > recHitsHandles(nSubDets_);
  std::vector< edm::ESHandle<HGCalGeometry> > geomHandle(nSubDets_);

  //read the generated primary vertex
  edm::Handle<edm::HepMCProduct> mcHandle;
  iEvent.getByToken(mc_, mcHandle);
  HepMC::GenVertex *primaryVertex = *(mcHandle)->GetEvent()->vertices_begin();
    
  ///CEE, HFE (Silicon), HBF (Scintillator)///
  for(size_t idet=0; idet<nSubDets_; ++idet) {
    iSetup.get<IdealGeometryRecord>().get(geometrySource_[idet], geomHandle[idet]);
  
    //collect rec hits in regions of interest
    iEvent.getByToken(recHitsTokens_[idet], recHitsHandles[idet]);
    int nMatched(0), nUnmatched(0);
    for(size_t i=0; i<recHitsHandles[idet]->size(); i++) {
      const HGCRecHit &h = recHitsHandles[idet]->operator[](i);
      const HGCalDetId did = h.detid();
    }
    
  tree_->Fill();
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalMaskResolutionAna);
