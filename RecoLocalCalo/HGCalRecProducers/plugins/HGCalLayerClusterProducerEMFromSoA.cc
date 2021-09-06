#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "CUDADataFormats/HGCal/interface/HGCCLUECPUHitsProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCCLUECPUClustersProduct.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCCLUESoA.h"

class HGCalLayerClusterProducerEMFromSoA : public edm::stream::EDProducer<> {
public:
  
  explicit HGCalLayerClusterProducerEMFromSoA(const edm::ParameterSet& ps);
  ~HGCalLayerClusterProducerEMFromSoA() override;
  void beginRun(edm::Run const&, edm::EventSetup const&);
  
  void produce(edm::Event&, const edm::EventSetup&) override;
  void getClusters_(uint32_t, uint32_t,
		    ConstHGCCLUEHitsSoA*, ConstHGCCLUEClustersSoA*,
		    reco::BasicClusterCollection&,
		    const HGCalDDDConstants*);

private:
  std::unique_ptr<reco::BasicClusterCollection> out_;
  reco::BasicClusterCollection clusters_;

  edm::ESHandle<CaloGeometry> geom_;
  const HGCalGeometry *geomEE_;
  std::pair<DetId::Detector, DetId::Detector> Det_;
  ForwardSubdetector SubDet_;
  const HGCalDDDConstants* ddd_ = nullptr;

  edm::EDGetTokenT<HGCCLUECPUHitsProduct> clueHitsSoAToken_;
  edm::EDGetTokenT<HGCCLUECPUClustersProduct> clueClustersSoAToken_;
  edm::EDPutTokenT<reco::BasicClusterCollection> clueCollectionToken_;
};

HGCalLayerClusterProducerEMFromSoA::HGCalLayerClusterProducerEMFromSoA(const edm::ParameterSet& ps) {
  clueHitsSoAToken_ = consumes<HGCCLUECPUHitsProduct>(ps.getParameter<edm::InputTag>("EMCLUEHitsSoATok")),
  clueClustersSoAToken_ = consumes<HGCCLUECPUClustersProduct>(ps.getParameter<edm::InputTag>("EMCLUEClustersSoATok"));
  clueCollectionToken_ = produces<reco::BasicClusterCollection>("Clusters");
}

HGCalLayerClusterProducerEMFromSoA::~HGCalLayerClusterProducerEMFromSoA() {}

void HGCalLayerClusterProducerEMFromSoA::beginRun(edm::Run const&, edm::EventSetup const& es)
{
  es.get<CaloGeometryRecord>().get(geom_);
  Det_ = std::make_pair(DetId::HGCalEE, DetId::HGCalHSi);
  SubDet_ = ForwardSubdetector::ForwardEmpty;
  const CaloSubdetectorGeometry* g = geom_->getSubdetectorGeometry(Det_.first, SubDet_);
  geomEE_ = dynamic_cast<const HGCalGeometry*>(g);
}

void HGCalLayerClusterProducerEMFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) {
  const HGCCLUECPUHitsProduct& clueHits = event.get(clueHitsSoAToken_);
  const HGCCLUECPUClustersProduct& clueClusters = event.get(clueClustersSoAToken_);
  ConstHGCCLUEHitsSoA clueHitsSoA = clueHits.get();
  ConstHGCCLUEClustersSoA clueClustersSoA = clueClusters.get();

  ddd_ = &(geomEE_->topology().dddConstants());
  
  out_ = std::make_unique<reco::BasicClusterCollection>();
  getClusters_(clueHits.nHits(), clueClusters.nClusters(),
	       &clueHitsSoA, &clueClustersSoA, *out_, ddd_);
  event.put(std::move(out_), "Clusters");
}

void HGCalLayerClusterProducerEMFromSoA::getClusters_(uint32_t nhits, uint32_t nclusters,
						      ConstHGCCLUEHitsSoA* hits,
                                                      ConstHGCCLUEClustersSoA* clusters,
						      reco::BasicClusterCollection& coll,
						      const HGCalDDDConstants* ddd) {
  coll.reserve(nclusters);

  for (unsigned i=0; i<nclusters; ++i) {

    if(i%5000==0) {
      std::cout << "WAFERZ: " << ddd->waferZ(clusters->layer[i], true) << std::endl;
      std::cout << "Layer: " << clusters->layer[i]  << std::endl;
     }
    math::XYZPoint position = math::XYZPoint(clusters->x[i],
					     clusters->y[i],
					     ddd->waferZ(clusters->layer[i], true) );


    //This code block is needed to match expected input from reco::BasicCluster
    std::vector<std::pair<DetId, float>> thisCluster;
    for (unsigned j=0; j<nhits; ++j) {
      if(hits->clusterIndex[j] == clusters->clusterIndex[i])
	thisCluster.emplace_back(hits->id[j], 1.f);
    }

    coll[i] = reco::BasicCluster(clusters->energy[i],
				 position,
				 reco::CaloID::DET_HGCAL_ENDCAP,
				 thisCluster,
				 reco::CaloCluster::hgcal_em); //reco::CaloCluster::hgcal_had for HAD section
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterProducerEMFromSoA);
