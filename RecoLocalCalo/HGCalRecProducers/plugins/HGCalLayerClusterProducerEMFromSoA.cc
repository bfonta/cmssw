#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

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
		    reco::BasicClusterCollection&);

private:
  std::unique_ptr<reco::BasicClusterCollection> out_;
  reco::BasicClusterCollection clusters_;

  edm::ESHandle<CaloGeometry> geom_;
  const HGCalGeometry *geomEE_;
  std::pair<DetId::Detector, DetId::Detector> Det_;
  ForwardSubdetector SubDet_;

  edm::EDGetTokenT<HGCCLUECPUHitsProduct> clueHitsSoAToken_;
  edm::EDGetTokenT<HGCCLUECPUClustersProduct> clueClustersSoAToken_;
  edm::EDPutTokenT<reco::BasicClusterCollection> clueCollectionToken_;

  //geometry
  hgcal::RecHitTools rhtools_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometry_token_;
};

HGCalLayerClusterProducerEMFromSoA::HGCalLayerClusterProducerEMFromSoA(const edm::ParameterSet& ps)
  : caloGeometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>())
{
  clueHitsSoAToken_ = consumes<HGCCLUECPUHitsProduct>(ps.getParameter<edm::InputTag>("EMCLUEHitsSoATok"));
  clueClustersSoAToken_ = consumes<HGCCLUECPUClustersProduct>(ps.getParameter<edm::InputTag>("EMCLUEClustersSoATok"));
  clueCollectionToken_ = produces<reco::BasicClusterCollection>("Clusters");

  //  caloGeomToken_(iC.esConsumes<CaloGeometry, CaloGeometryRecord>())
}

HGCalLayerClusterProducerEMFromSoA::~HGCalLayerClusterProducerEMFromSoA() {}

void HGCalLayerClusterProducerEMFromSoA::beginRun(edm::Run const&, edm::EventSetup const& es)
{
  es.get<CaloGeometryRecord>().get(geom_);
  Det_ = std::make_pair(DetId::HGCalEE, DetId::HGCalHSi);
  SubDet_ = ForwardSubdetector::ForwardEmpty;
  const CaloSubdetectorGeometry* g = geom_->getSubdetectorGeometry(Det_.first, SubDet_);
  geomEE_ = dynamic_cast<const HGCalGeometry*>(g);

  const CaloGeometry& geom = es.getData(caloGeometry_token_);
  rhtools_.setGeometry(geom);
}

void HGCalLayerClusterProducerEMFromSoA::produce(edm::Event& event, const edm::EventSetup& setup) { 
  const HGCCLUECPUHitsProduct& clueHits = event.get(clueHitsSoAToken_);
  const HGCCLUECPUClustersProduct& clueClusters = event.get(clueClustersSoAToken_);
  ConstHGCCLUEHitsSoA clueHitsSoA = clueHits.get();
  ConstHGCCLUEClustersSoA clueClustersSoA = clueClusters.get();
  
  out_ = std::make_unique<reco::BasicClusterCollection>();

  getClusters_(clueHits.nHits(), clueClusters.nClusters(),
	       &clueHitsSoA, &clueClustersSoA, *out_);

  event.put(std::move(out_), "Clusters");
}

void HGCalLayerClusterProducerEMFromSoA::getClusters_(uint32_t nhits, uint32_t nclusters,
						      ConstHGCCLUEHitsSoA* hits,
                                                      ConstHGCCLUEClustersSoA* clusters,
						      reco::BasicClusterCollection& coll) {
  coll.reserve(nclusters);
  for (unsigned i=0; i<nclusters; ++i) {

    if( clusters->energy[i] > 0.) { //get rid of excess empty GPU clusters

      math::XYZPoint position = math::XYZPoint(clusters->x[i],
					       clusters->y[i],
					       //rhtools_.getPosition( 2416969935 ).z() );
					       rhtools_.getPosition( clusters->seedId[i] ).z() );

      //This code block is needed to match expected input from reco::BasicCluster
      // 
      // for (unsigned j=0; j<nhits; ++j) {
      //   if(hits->clusterIndex[j] == clusters->clusterIndex[i])
      // 	thisCluster.emplace_back(hits->id[j], 1.f);
      // }
      std::vector<std::pair<DetId, float>> thisCluster;

      coll.emplace_back( clusters->energy[i],
			 position,
			 reco::CaloID::DET_HGCAL_ENDCAP,
			 thisCluster,
			 reco::CaloCluster::hgcal_em //reco::CaloCluster::hgcal_had for HAD section
			 );
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterProducerEMFromSoA);
