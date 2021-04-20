#include <fstream>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/MessageDefinition.pb.h"

class BinaryWriter : public edm::EDAnalyzer 
{  
 public:
  
  explicit BinaryWriter( const edm::ParameterSet& );
  ~BinaryWriter();
  void fillEventProtocol(const HGCeeUncalibratedRecHitCollection&, uncalibRecHitsProtocol::Event*);
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();

 private:
  uncalibRecHitsProtocol::Data data_;
  edm::EDGetTokenT<HGCeeUncalibratedRecHitCollection> uncalibRecHitCPUToken_;
  std::string fileName_;
};

BinaryWriter::BinaryWriter(const edm::ParameterSet &ps):
  uncalibRecHitCPUToken_{consumes<HGCUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEUncalibRecHitsTok"))},
  fileName_{ps.getParameter<std::string>("fileName")}
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

BinaryWriter::~BinaryWriter()
{
  // Write the new address book back to disk.
  std::ofstream output(fileName_, std::ios::out);
  if (!data_.SerializeToOstream(&output)) {
    edm::LogError("ParseError") << "Failed to write.";
  }
  
  google::protobuf::ShutdownProtobufLibrary();
}


void BinaryWriter::endJob()
{
}

void BinaryWriter::fillEventProtocol(const HGCeeUncalibratedRecHitCollection& src, uncalibRecHitsProtocol::Event* dst) {
  dst->set_integer(2);
}

void BinaryWriter::analyze( const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  const auto& hits = iEvent.get(uncalibRecHitCPUToken_);
  const unsigned nhits(hits.size());
  
  if (nhits == 0)
    edm::LogError("BinaryWriter") << "WARNING: no input hits!";

  // Add an event to the protocol buffer.
  fillEventProtocol(hits, data_.add_events());
}

DEFINE_FWK_MODULE(BinaryWriter);
