#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUECPUClustersProduct.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"

class HGCalLayerClusterProducerEMFromSoA : public edm::stream::EDProducer<> {
public:
  
  explicit HGCalLayerClusterProducerEMFromSoA(const edm::ParameterSet& ps);
  ~HGCalLayerClusterProducerEMFromSoA() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void getClusters_(uint32_t, reco::BasicClusterCollection&, ConstHGCCLUEClustersSoA*);

private:
  std::unique_ptr<reco::BasicClusterCollection> out_;
  reco::BasicClusterCollection clusters_;

  edm::EDGetTokenT<HGCCLUECPUClustersProduct> clueSoAToken_;
  edm::EDPutTokenT<reco::BasicClusterCollection> clueCollectionToken_;
};

HGCalLayerClusterProducerEMFromSoA::HGCalLayerClusterProducerEMFromSoA(const edm::ParameterSet& ps) {
  clueSoAToken_ = consumes<HGCCLUECPUClustersProduct>(ps.getParameter<edm::InputTag>("EECLUESoATok"));
  clueCollectionToken_ = produces<reco::BasicClusterCollection>();
}

HGCalLayerClusterProducerEMFromSoA::~HGCalLayerClusterProducerEMFromSoA() {}

void HGCalLayerClusterProducerEMFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  const HGCCLUECPUClustersProduct& clueClusters = event.get(clueSoAToken_);
  ConstHGCCLUEClustersSoA clueSoA = clueClusters.get();

  out_ = std::make_unique<reco::BasicClusterCollection>();
  getClusters_(clueClusters.nClusters(), *out_, &clueSoA);
  event.put(std::move(out_));
}

void HGCalLayerClusterProducerEMFromSoA::getClusters_(uint32_t nclusters,
                                                      reco::BasicClusterCollection& out,
                                                      ConstHGCCLUEClustersSoA* soa) {
  clusters_.reserve(nclusters);

  // for (uint i = 0; i < nclusters; ++i) {
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
