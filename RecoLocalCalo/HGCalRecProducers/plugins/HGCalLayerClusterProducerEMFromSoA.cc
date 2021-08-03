#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUECPUProduct.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"

class HGCalLayerClusterProducerEMFromSoA : public edm::stream::EDProducer<> {
public:
  
  explicit HGCalLayerClusterProducerEMFromSoA(const edm::ParameterSet& ps);
  ~HGCalLayerClusterProducerEMFromSoA() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void getClusters_(uint32_t, reco::BasicClusterCollection&, ConstHGCCLUESoA*);

private:
  std::unique_ptr<reco::BasicClusterCollection> out_;
  reco::BasicClusterCollection clusters_;

  edm::EDGetTokenT<HGCCLUECPUProduct> clueSoAToken_;
  edm::EDPutTokenT<reco::BasicClusterCollection> clueCollectionToken_;
};

HGCalLayerClusterProducerEMFromSoA::HGCalLayerClusterProducerEMFromSoA(const edm::ParameterSet& ps) {
  clueSoAToken_ = consumes<HGCCLUECPUProduct>(ps.getParameter<edm::InputTag>("EECLUESoATok"));
  clueCollectionToken_ = produces<reco::BasicClusterCollection>();
}

HGCalLayerClusterProducerEMFromSoA::~HGCalLayerClusterProducerEMFromSoA() {}

void HGCalLayerClusterProducerEMFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  const HGCCLUECPUProduct& clueHits = event.get(clueSoAToken_);
  ConstHGCCLUESoA clueSoA = clueHits.get();

  out_ = std::make_unique<reco::BasicClusterCollection>();
  getClusters_(clueHits.nHits(), *out_, &clueSoA);
  event.put(std::move(out_));
}

void HGCalLayerClusterProducerEMFromSoA::getClusters_(uint32_t nhits,
                                                      reco::BasicClusterCollection& out,
                                                      ConstHGCCLUESoA* soa) {
  clusters_.reserve(nhits);

  // for (uint i = 0; i < nhits; ++i) {
  //   DetId id_converted(soa->id[i]);
  //   float son = soa->energy[i]/soa->sigmaNoise[i];
  //   cluehits.emplace_back(id_converted,
  //                        soa->energy[i],
  //                        soa->time[i],
  //                        0,
  //                        soa->flagBits[i],
  //                        son,
  //                        soa->timeError[i]);
  //   cluehits[i].setSignalOverSigmaNoise(son);
  // }
  //TO BE DONE!!!!!
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterProducerEMFromSoA);
