#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include <iostream>
#include <array>
#include "TTree.h"

class EcalGeometryAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EcalGeometryAnalyzer(const edm::ParameterSet&);
  ~EcalGeometryAnalyzer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}
  double inBarrel(const DetId& id);

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  edm::EDGetTokenT<reco::PFRecHitCollection> recHitToken_;
  edm::EDGetTokenT<std::vector<PCaloHit>> simHitToken_;
  edm::EDGetTokenT<reco::PFClusterCollection> recClusterToken_;
  edm::EDGetTokenT<SimClusterCollection> simClusterToken_;

  TTree *geomTree_, *eventTree_;

  unsigned crystalDetId_;
  float crystalCenterEta_;
  float crystalCenterPhi_;
  float crystalCorner0Eta_;
  float crystalCorner1Eta_;
  float crystalCorner2Eta_;
  float crystalCorner3Eta_;
  float crystalCorner0Phi_;
  float crystalCorner1Phi_;
  float crystalCorner2Phi_;
  float crystalCorner3Phi_;

  static constexpr std::array<std::string, 2> prefixes_ = {{"Reco", "Sim"}};
  unsigned eventId_;

  template <typename T>
  using UMap = std::unordered_map<std::string, T>;
								  
  UMap<unsigned> nHits_;
  UMap<std::vector<float>> energies_;
  UMap<std::vector<unsigned>> detids_;

  UMap<std::vector<float>> clusterEnergies_;
  UMap<std::vector<float>> clusterEtas_;
  UMap<std::vector<float>> clusterPhis_;
  UMap<std::vector<unsigned>> clusterHitDetids_;
  UMap<std::vector<unsigned>> clusterHitClids_;
  UMap<std::vector<float>> clusterHitEnergies_;
  UMap<std::vector<float>> clusterHitFractions_;
};

EcalGeometryAnalyzer::EcalGeometryAnalyzer(const edm::ParameterSet& iConfig)
  : caloGeomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
	recHitToken_(consumes<reco::PFRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHits"))),
	simHitToken_(consumes<std::vector<PCaloHit>>(iConfig.getParameter<edm::InputTag>("simHits"))),
	recClusterToken_(consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("recClusters"))),
	simClusterToken_(consumes<SimClusterCollection>(iConfig.getParameter<edm::InputTag>("simClusters"))) {
  edm::Service<TFileService> fs;
  geomTree_ = fs->make<TTree>("Geometry", "Geometry data");
  eventTree_ = fs->make<TTree>("Event", "Event data");
}
// hltParticleFlowClusterECALUnseeded
void EcalGeometryAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHits", edm::InputTag("hltParticleFlowRecHitECALUnseeded"));
  desc.add<edm::InputTag>("simHits", edm::InputTag("g4SimHits", "EcalHitsEB"));
  desc.add<edm::InputTag>("recClusters", edm::InputTag("hltParticleFlowClusterECAL"));
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));    
  descriptions.add("ecalGeometryAnalyzer", desc);
}

void EcalGeometryAnalyzer::beginJob() {
  geomTree_->Branch("crystalDetId", &crystalDetId_);
  geomTree_->Branch("crystalCenterEta", &crystalCenterEta_);
  geomTree_->Branch("crystalCenterPhi", &crystalCenterPhi_);
  geomTree_->Branch("crystalCorner0Eta", &crystalCorner0Eta_);
  geomTree_->Branch("crystalCorner1Eta", &crystalCorner1Eta_);
  geomTree_->Branch("crystalCorner2Eta", &crystalCorner2Eta_);
  geomTree_->Branch("crystalCorner3Eta", &crystalCorner3Eta_);
  geomTree_->Branch("crystalCorner0Phi", &crystalCorner0Phi_);
  geomTree_->Branch("crystalCorner1Phi", &crystalCorner1Phi_);
  geomTree_->Branch("crystalCorner2Phi", &crystalCorner2Phi_);
  geomTree_->Branch("crystalCorner3Phi", &crystalCorner3Phi_);

  eventTree_->Branch("eventId", &eventId_);

  for (auto& prefix : prefixes_) {
	eventTree_->Branch(("nHits" + prefix).c_str(), &nHits_[prefix]);
	eventTree_->Branch(("energies" + prefix).c_str(), &energies_[prefix]);
	eventTree_->Branch(("detids" + prefix).c_str(), &detids_[prefix]);

	eventTree_->Branch(("clusterEnergies" + prefix).c_str(), &clusterEnergies_[prefix]);
	eventTree_->Branch(("clusterEtas" + prefix).c_str(), &clusterEtas_[prefix]);
	eventTree_->Branch(("clusterPhis" + prefix).c_str(), &clusterPhis_[prefix]);
	eventTree_->Branch(("clusterHitDetids" + prefix).c_str(), &clusterHitDetids_[prefix]);
	eventTree_->Branch(("clusterHitClids" + prefix).c_str(), &clusterHitClids_[prefix]);
	eventTree_->Branch(("clusterHitEnergies" + prefix).c_str(), &clusterHitEnergies_[prefix]);
	eventTree_->Branch(("clusterHitFractions" + prefix).c_str(), &clusterHitFractions_[prefix]);
  }
}

// check the detid lies in the ECAL barrel
double EcalGeometryAnalyzer::inBarrel(const DetId& id) {
  return id.det() == DetId::Ecal && id.subdetId() == EcalBarrel;
}

void EcalGeometryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the ECAL geometry
  const auto& caloGeom = iSetup.getData(caloGeomToken_);
  const auto& barrelGeom = caloGeom.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const std::vector<DetId> detids = barrelGeom->getValidDetIds();

  unsigned eventId = iEvent.id().event();
  eventId_ = eventId;

  // Reset vector variables
  for (auto& prefix : prefixes_) {
	detids_[prefix].clear();
	energies_[prefix].clear();

	clusterEnergies_[prefix].clear();
	clusterEtas_[prefix].clear();
	clusterPhis_[prefix].clear();
	clusterHitEnergies_[prefix].clear();
	clusterHitFractions_[prefix].clear();
	clusterHitDetids_[prefix].clear();
	clusterHitClids_[prefix].clear();
  }
  
  // Geometry fill
  if (eventId == 1) {
    for (auto& did : detids) {
      if (did.subdetId() != EcalBarrel) {
        throw std::runtime_error("Error");
        continue;
      }

      const CaloCellGeometry* cellGeom = barrelGeom->getGeometry(did);
      crystalDetId_ = did.rawId();
      crystalCenterEta_ = cellGeom->getPosition().eta();
      crystalCenterPhi_ = cellGeom->getPosition().phi();

      const EZArrayFL<GlobalPoint> corners = cellGeom->getCorners();
      crystalCorner0Eta_ = corners[0].eta();
      crystalCorner1Eta_ = corners[1].eta();
      crystalCorner2Eta_ = corners[2].eta();
      crystalCorner3Eta_ = corners[3].eta();
      crystalCorner0Phi_ = corners[0].phi();
      crystalCorner1Phi_ = corners[1].phi();
      crystalCorner2Phi_ = corners[2].phi();
      crystalCorner3Phi_ = corners[3].phi();

      geomTree_->Fill();
    }
  }  // if (eventId == 1)
  
  edm::Handle<reco::PFRecHitCollection> recHits_;
  iEvent.getByToken(recHitToken_, recHits_);
  if (!recHits_.isValid()) {
    edm::LogInfo("EcalGeometryAnalyzer") << "Input recHit collection not found.";
    return;
  }
  edm::Handle<std::vector<PCaloHit>> simHits_;
  iEvent.getByToken(simHitToken_, simHits_);
  if (!simHits_.isValid()) {
    edm::LogInfo("EcalGeometryAnalyzer") << "Input simHit collection not found.";
    return;
  }
  edm::Handle<reco::PFClusterCollection> recClusters_;
  iEvent.getByToken(recClusterToken_, recClusters_);
  if (!recClusters_.isValid()) {
    edm::LogInfo("EcalGeometryAnalyzer") << "Input recCluster collection not found.";
    return;
  }
  edm::Handle<SimClusterCollection> simClusters_;
  iEvent.getByToken(simClusterToken_, simClusters_);
  if (!simClusters_.isValid()) {
    edm::LogInfo("EcalGeometryAnalyzer") << "Input simCluster collection not found.";
    return;
  }

  auto recHits = *recHits_;
  auto simHits = *simHits_;
  auto recClusters = *recClusters_;
  auto simClusters = *simClusters_;

  // Event fill
  nHits_["Reco"] = recHits.size();
  for (auto& rechit : recHits) {
    DetId id(rechit.detId());
    if (!inBarrel(id))
      continue;
    detids_["Reco"].push_back(rechit.detId());
    energies_["Reco"].push_back(rechit.energy());
  }

  nHits_["Sim"] = simHits.size();
  for (auto& simhit : simHits) {
    DetId id(simhit.id());
    if (!inBarrel(id))
      continue;
    detids_["Sim"].push_back(simhit.id());
    energies_["Sim"].push_back(simhit.energy());
  }

  // reco clusters
  unsigned clusterCounter = 0;
  for (auto& cl : recClusters) {
	clusterCounter++;
	
	// properties of the clusters
	clusterEnergies_["Reco"].push_back(cl.energy());
	clusterEtas_["Reco"].push_back(cl.eta());
	clusterPhis_["Reco"].push_back(cl.phi());

	for (auto const& rechit : cl.recHitFractions()) {
	  const auto& ref = rechit.recHitRef();
	  DetId clhitId(ref->detId());
	  if (!inBarrel(clhitId))
		continue;

	  clusterHitDetids_["Reco"].push_back(clhitId);
	  clusterHitClids_["Reco"].push_back(clusterCounter);
	  clusterHitEnergies_["Reco"].push_back(ref->energy());
	  clusterHitFractions_["Reco"].push_back(rechit.fraction());
	}

	// // properties of the hits in each cluster
	// for (auto [clhitId, clhitFrac] : cl.hitsAndFractions()) {
	//   if (!inBarrel(clhitId))
	// 	continue;

	//   for (auto& rechit : recHits) {
	// 	if (clhitId == rechit.detId()) {
	// 	  float energy = rechit.energy();
	// 	  clusterHitDetids_["Reco"].push_back(clhitId);
	// 	  clusterHitClids_["Reco"].push_back(clusterCounter);
	// 	  clusterHitEnergies_["Reco"].push_back(energy);
	// 	  clusterHitFractions_["Reco"].push_back(clhitFrac);
	// 	  break;
	// 	}
	//   }
	// }
  }

  // sim clusters
  clusterCounter = 0;
  for (auto& cl : simClusters) {
	clusterCounter++;
	
	// properties of the clusters
	clusterEnergies_["Sim"].push_back(cl.energy());
	clusterEtas_["Sim"].push_back(cl.eta());
	clusterPhis_["Sim"].push_back(cl.phi());
	
	// properties of the hits in each cluster
	const auto& hits_fractions = cl.hits_and_fractions();
	const auto& hits_energies = cl.hits_and_energies();
	
	auto itF = hits_fractions.begin();
	auto itE = hits_energies.begin();
	for (; itF != hits_fractions.end() && itE != hits_energies.end(); ++itF, ++itE) {
	  DetId clhitId(itF->first);
	  if (!inBarrel(clhitId))
		continue;
	  clusterHitDetids_["Sim"].push_back(clhitId);
	  clusterHitClids_["Sim"].push_back(clusterCounter);
	  clusterHitEnergies_["Sim"].push_back(itE->second);
	  clusterHitFractions_["Sim"].push_back(itF->second);
	}
  }
  

  eventTree_->Fill();
}

DEFINE_FWK_MODULE(EcalGeometryAnalyzer);
