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
};

BinaryReader::BinaryReader(const edm::ParameterSet &ps):
  fileName_{ps.getParameter<std::string>("fileName")}
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::ifstream input(fileName_, std::ios::in);
  if (!data_.ParseFromIstream(&input)) {
    edm::LogError("ParseError") << "Failed to parse.";
  }
}

BinaryReader::~BinaryReader()
{
  google::protobuf::ShutdownProtobufLibrary();
}


void BinaryReader::endJob()
{
}

void BinaryReader::listEventProtocol(const uncalibRecHitsProtocol::Data& src) {
  std::cout << src.events_size() << std::endl;
  /*
  for (int i = 0; i < src.people_size(); i++) {
    const tutorial::Person& person = address_book.people(i);

    cout << "Person ID: " << person.id() << endl;
    cout << "  Name: " << person.name() << endl;
    if (person.has_email()) {
      cout << "  E-mail address: " << person.email() << endl;
    }

    for (int j = 0; j < person.phones_size(); j++) {
      const tutorial::Person::PhoneNumber& phone_number = person.phones(j);

      switch (phone_number.type()) {
        case tutorial::Person::MOBILE:
          cout << "  Mobile phone #: ";
          break;
        case tutorial::Person::HOME:
          cout << "  Home phone #: ";
          break;
        case tutorial::Person::WORK:
          cout << "  Work phone #: ";
          break;
      }
      cout << phone_number.number() << endl;
    }
  }
  */
}

void BinaryReader::analyze( const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
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
