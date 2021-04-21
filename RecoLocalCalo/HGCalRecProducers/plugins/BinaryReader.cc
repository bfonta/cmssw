#include <fstream>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/MessageDefinition.pb.h"

class BinaryReader : public edm::EDAnalyzer 
{  
 public:
  
  explicit BinaryReader( const edm::ParameterSet& );
  ~BinaryReader();
  void listEventProtocol(const uncalibRecHitsProtocol::Data&);
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();

 private:
  uncalibRecHitsProtocol::Data data_;
  edm::EDGetTokenT<HGCeeUncalibratedRecHitCollection> uncalibRecHitCPUToken_;
  std::string fileName_;
  unsigned nEvents_;

  unsigned counter_;
};

BinaryReader::BinaryReader(const edm::ParameterSet &ps):
  fileName_{ps.getParameter<std::string>("fileName")},
  nEvents_{ps.getParameter<unsigned>("nEvents")}
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::ifstream input(fileName_, std::ios::in);
  if (!data_.ParseFromIstream(&input)) {
    edm::LogError("ParseError") << "Failed to parse.";
  }

  counter_ = 0;
}

BinaryReader::~BinaryReader()
{
  google::protobuf::ShutdownProtobufLibrary();
}


void BinaryReader::endJob()
{
}

void BinaryReader::listEventProtocol(const uncalibRecHitsProtocol::Data& src)
{
  unsigned dataEntry = counter_ % nEvents_;
  const uncalibRecHitsProtocol::Event& ev = data_.events( dataEntry );
  std::cout << src.events_size() << ", " << ev.amplitude_size() << ", " << counter_ << std::endl;
  std::cout << "Amplitude " << ev.amplitude(2) << std::endl;
}

void BinaryReader::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  ++counter_;
  /*
  const auto& hits = iEvent.get(uncalibRecHitCPUToken_);
  const unsigned nhits(hits.size());
  
  if (nhits == 0)
    edm::LogError("BinaryReader") << "WARNING: no input hits!";
  */

  // Add an event to the protocol buffer.
  listEventProtocol(data_);
}

DEFINE_FWK_MODULE(BinaryReader);
