#ifndef _HGCalMaskResolutionAna_h_
#define _HGCalMaskResolutionAna_h_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TTree.h"
#include "TH1F.h"

#include <iostream>
#include <string>

class HeterogeneousHGCalRecHitsValidator : public edm::EDAnalyzer 
{
 public:
  explicit HeterogeneousHGCalRecHitsValidator( const edm::ParameterSet& );
  ~HeterogeneousHGCalRecHitsValidator();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();

 private:
  edm::EDGetTokenT<edm::HepMCProduct> mc_;
  std::vector< edm::EDGetTokenT<HGCRecHitCollection> > recHitsTokens_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticles_;

  hgcal::RecHitTools recHitTools_;

  std::vector< TH1F* > zhist;
  std::vector< TH1F* > xhist;
  std::vector< TH1F* > yhist; 
  std::vector< TH1F* > zsidehist;
  std::vector< TH1F* > ehist;
  std::vector< TH1F* > layerhist;
  std::vector< TH1F* > offsetlayerhist;

  TTree* tree_;
  std::string treename_;
};
 

#endif
