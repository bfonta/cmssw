#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

class DigisTester : public DQMEDAnalyzer {
public:
  explicit DigisTester(const edm::ParameterSet&);  

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  edm::EDGetTokenT<EBDigiCollection> ecalEBDigisToken_;
  edm::EDGetTokenT<EEDigiCollection> ecalEEDigisToken_;

  std::string outFolder_;

  using U2Map = std::unordered_map<std::string, MonitorElement*>;
  U2Map h2d_ebdigis_, h2d_eedigis_;
};

DigisTester::DigisTester(const edm::ParameterSet& iConfig)
  : geometry_token_(esConsumes()),
	ecalEBDigisToken_(consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalEBDigis"))),
	ecalEEDigisToken_(consumes<EEDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalEEDigis"))),
	outFolder_(iConfig.getParameter<std::string>("outFolder")) {}

void DigisTester::bookHistograms(DQMStore::IBooker& ibook,
								 edm::Run const&,
								 edm::EventSetup const&) {
  ibook.setCurrentFolder(outFolder_ + "/Digis");

  unsigned mNBinsEn = 6000;
  float mEnMin = 3000.;
  float mEnMax = 6000.;
  unsigned mNBinsMult = 400;
  float mMultMax = 400.;
  unsigned mNBinsPhi = 360;
  float mPhiMax = 360.;
  unsigned mNBinsEta = 100;
  float mEtaMax = 3.3;

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> ebVars = {
	{"En_Eta",   std::make_tuple(mNBinsEn, mEnMin, mEnMax, mNBinsEta, -mEtaMax, mEtaMax)},
	{"En_Phi",   std::make_tuple(mNBinsEn, mEnMin, mEnMax, mNBinsPhi, 0, mPhiMax)},
	{"En_Mult",  std::make_tuple(mNBinsEn, mEnMin, mEnMax, mNBinsMult, 0., mMultMax)},
	{"Mult_Eta", std::make_tuple(mNBinsMult, 0., mMultMax, mNBinsEta, -mEtaMax, mEtaMax)},
	{"Mult_Phi", std::make_tuple(mNBinsMult, 0., mMultMax, mNBinsPhi, 0, mPhiMax)},
  };

  for (auto& ebVar : ebVars) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = ebVar.second;
    auto x_title = ebVar.first.substr(0, ebVar.first.find("_"));
    auto y_title = ebVar.first.substr(ebVar.first.find("_") + 1);
    h2d_ebdigis_[ebVar.first] = ibook.book2D("EcalEBDigis" + ebVar.first,
											 "EcalEBDigis;" + x_title + ";" + y_title,
											 nBinsX, hMinX, hMaxX,
											 nBinsY, hMinY, hMaxY);
  }

  unsigned mNBinsX = 100;
  float mXMax = 100.;
  unsigned mNBinsY = 100;
  float mYMax = 100.;

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> eeVars = {
	{"Digi_X",   std::make_tuple(mNBinsEn, mEnMin, mEnMax, mNBinsX, -mXMax, mXMax)},
	{"Digi_Y",   std::make_tuple(mNBinsEn, mEnMin, mEnMax, mNBinsY, 0., mYMax)},
	{"ADC_X", std::make_tuple(mNBinsMult, 0., mMultMax, mNBinsX, 0., mXMax)},
	{"ADC_Y", std::make_tuple(mNBinsMult, 0., mMultMax, mNBinsY, 0., mYMax)},
  };

  for (auto& eeVar : eeVars) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = eeVar.second;
    auto x_title = eeVar.first.substr(0, eeVar.first.find("_"));
    auto y_title = eeVar.first.substr(eeVar.first.find("_") + 1);
    h2d_eedigis_[eeVar.first] = ibook.book2D("EcalEEDigis" + eeVar.first,
											 "EcalEEDigis;" + x_title + ";" + y_title,
											 nBinsX, hMinX, hMaxX,
											 nBinsY, hMinY, hMaxY);
  }
}

void DigisTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<EBDigiCollection> ebDigisHandle;
  iEvent.getByToken(ecalEBDigisToken_, ebDigisHandle);
  if (!ebDigisHandle.isValid()) {
    edm::LogPrint("DigisTester") << "Input EB Digis collection not found.";
    return;
  }

  edm::Handle<EEDigiCollection> eeDigisHandle;
  iEvent.getByToken(ecalEEDigisToken_, eeDigisHandle);
  if (!eeDigisHandle.isValid()) {
    edm::LogPrint("DigisTester") << "Input EE Digis collection not found.";
    return;
  }

  for (const edm::DataFrame& digi : *ebDigisHandle) {
    EBDetId ebId(digi.id());

    // Get the crystal's ieta and iphi
    int ieta = ebId.ieta();
    int iphi = ebId.iphi();
    float eta = ebId.approxEta(); // return an approximate values of eta (~0.15% precise)
	
    // Get the number of ADC samples (usually 10)
    int nADCSamples = digi.size();

	float digiMax = 0;
	float adcMax = 0;
	for (int i=0; i<nADCSamples; ++i) {
	  float adcVal = static_cast<EBDataFrame>(digi).sample(i).adc();
	  if (adcVal > adcMax) adcMax = adcVal;
	  float digiVal = digi[i];
	  if (digiVal > digiMax) digiMax = digiVal;
	}
	
	h2d_ebdigis_["En_Eta"]->Fill(digiMax, eta);
	h2d_ebdigis_["En_Phi"]->Fill(digiMax, iphi);
	h2d_ebdigis_["En_Mult"]->Fill(digiMax, adcMax);
	h2d_ebdigis_["Mult_Eta"]->Fill(adcMax, eta);
	h2d_ebdigis_["Mult_Phi"]->Fill(adcMax, iphi);
  }

  for (const edm::DataFrame& digi : *eeDigisHandle) {
    EEDetId eeId(digi.id());

    int ix = eeId.ix();
    int iy = eeId.iy();

	    // Get the number of ADC samples (usually 10)
    int nADCSamples = digi.size();

	float digiMax = 0;
	float adcMax = 0;
	for (int i=0; i<nADCSamples; ++i) {
	  float adcVal = static_cast<EEDataFrame>(digi).sample(i).adc();
	  if (adcVal > adcMax) adcMax = adcVal;
	  float digiVal = digi[i];
	  if (digiVal > digiMax) digiMax = digiVal;
	}

	h2d_eedigis_["Digi_X"]->Fill(digiMax, ix);
	h2d_eedigis_["Digi_Y"]->Fill(digiMax, iy);
	h2d_eedigis_["ADC_X"]->Fill(adcMax, ix);
	h2d_eedigis_["ADC_Y"]->Fill(adcMax, iy);
  }
}

DEFINE_FWK_MODULE(DigisTester);
