#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUECPUProduct.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"

using OutColl = ConstHGCCLUESoA; //CHANGE!!!

class LayerClusterProducerEMFromSoA : public edm::stream::EDProducer<> {
public:
  
  explicit LayerClusterProducerEMFromSoA(const edm::ParameterSet& ps);
  ~LayerClusterProducerEMFromSoA() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void convert_soa_data_to_collection_(uint32_t, OutColl&, ConstHGCCLUESoA*);

private:
  std::unique_ptr<OutColl> cluehits_;
  edm::EDGetTokenT<HGCCLUECPUProduct> clueSoAToken_;
  edm::EDPutTokenT<OutColl> clueCollectionToken_;
};

LayerClusterProducerEMFromSoA::LayerClusterProducerEMFromSoA(const edm::ParameterSet& ps) {
  clueSoAToken_ = consumes<HGCCLUECPUProduct>(ps.getParameter<edm::InputTag>("EECLUESoATok"));
  clueCollectionToken_ = produces<OutColl>();
}

LayerClusterProducerEMFromSoA::~LayerClusterProducerEMFromSoA() {}

void LayerClusterProducerEMFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  const HGCCLUECPUProduct& clueHits = event.get(clueSoAToken_);
  ConstHGCCLUESoA clueSoA = clueHits.get();
  cluehits_ = std::make_unique<OutColl>();
  convert_soa_data_to_collection_(clueHits.nHits(), *cluehits_, &clueSoA);
  event.put(std::move(cluehits_));
}

void LayerClusterProducerEMFromSoA::convert_soa_data_to_collection_(uint32_t nhits,
                                                      OutColl& cluehits,
                                                      ConstHGCCLUESoA* soa) {
  // cluehits.reserve(nhits);
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
  printf("TO BE DEFINED!!!!!\n");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LayerClusterProducerEMFromSoA);
