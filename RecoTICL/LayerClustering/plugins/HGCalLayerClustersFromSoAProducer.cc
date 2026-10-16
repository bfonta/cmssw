#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/HGCalReco/interface/BarrelSoARecHitsExtraHostCollection.h"
#include "DataFormats/HGCalReco/interface/BarrelSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtraHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"

#define DEBUG_CLUSTERS_ALPAKA 0

#if DEBUG_CLUSTERS_ALPAKA
#include "RecoTICL/LayerClustering/interface/DumpClustersDetails.h"
#endif

namespace {

  // HGCAL: EE, FH, BH and HFNose
  struct HGCalLayerClustersFromSoATraits {
    using RecHitsHostCollection = HGCalSoARecHitsHostCollection;
    using RecHitsExtraHostCollection = HGCalSoARecHitsExtraHostCollection;
    static constexpr const char* recHitsSoA = "hgcalRecHitsSoA";
    static constexpr const char* recHitsLayerClustersSoA = "hgcalRecHitsLayerClustersSoA";

    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<edm::InputTag>("src", edm::InputTag("hltHgcalSoALayerClustersProducer"));
      desc.add<edm::InputTag>(recHitsLayerClustersSoA, edm::InputTag("hltHgcalSoARecHitsLayerClustersProducer"));
      desc.add<edm::InputTag>(recHitsSoA, edm::InputTag("hltHgcalSoARecHitsProducer"));
      desc.ifValue(edm::ParameterDescription<std::string>(
                       "detector", "EE", true, edm::Comment("the HGCAL component used to create clusters.")),
                   edm::allowedValues<std::string>("EE", "FH", "BH", "HFNose"));
    }

    static reco::CaloCluster::AlgoId algoId(const std::string& detector) {
      if (detector == "HFNose") {
        return reco::CaloCluster::hfnose;
      } else if (detector == "EE") {
        return reco::CaloCluster::hgcal_em;
      } else if (detector == "BH") {
        return reco::CaloCluster::hgcal_scintillator;
      }
      return reco::CaloCluster::hgcal_had;  // for FH
    }

    static reco::CaloID::Detectors caloID(const std::string&) { return reco::CaloID::DET_HGCAL_ENDCAP; }

    // The layerClusterMask for the HGCAL detector is created at a later stage, when the layer clusters from the
    // different components of HGCAL are merged together into a unique collection. For the case of HFNose, since there
    // is no further merging step needed, we create the layerClustersMask directly here.
    static bool producesMask(const std::string& detector) { return detector == "HFNose"; }
  };

  // ECAL (EB) and HCAL (HB) barrel
  struct BarrelLayerClustersFromSoATraits {
    using RecHitsHostCollection = BarrelSoARecHitsHostCollection;
    using RecHitsExtraHostCollection = BarrelSoARecHitsExtraHostCollection;
    static constexpr const char* recHitsSoA = "barrelRecHitsSoA";
    static constexpr const char* recHitsLayerClustersSoA = "barrelRecHitsLayerClustersSoA";

    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<edm::InputTag>("src", edm::InputTag("barrelSoALayerClustersEB"));
      desc.add<edm::InputTag>(recHitsLayerClustersSoA, edm::InputTag("barrelCLUEsteringEB"));
      desc.add<edm::InputTag>(recHitsSoA, edm::InputTag("barrelSoARecHitsEB"));
      desc.ifValue(edm::ParameterDescription<std::string>(
                       "detector", "EB", true, edm::Comment("the barrel calorimeter: EB (ECAL) or HB (HCAL)")),
                   edm::allowedValues<std::string>("EB", "HB"));
    }

    static reco::CaloCluster::AlgoId algoId(const std::string& detector) {
      return detector == "EB" ? reco::CaloCluster::barrel_em : reco::CaloCluster::barrel_had;
    }

    static reco::CaloID::Detectors caloID(const std::string& detector) {
      return detector == "EB" ? reco::CaloID::DET_ECAL_BARREL : reco::CaloID::DET_HCAL_BARREL;
    }

    // Like BarrelLayerClusterProducer, which the barrel layer clusters from the SoA replace
    static bool producesMask(const std::string&) { return true; }
  };

}  // namespace

// Converts the layer clusters SoA of one detector back to legacy reco::BasicCluster products
template <typename Traits>
class LayerClustersFromSoAProducerT : public edm::stream::EDProducer<> {
public:
  LayerClustersFromSoAProducerT(edm::ParameterSet const& config)
      : getTokenSoAClusters_(consumes(config.getParameter<edm::InputTag>("src"))),
        getTokenSoACells_(consumes(config.getParameter<edm::InputTag>(Traits::recHitsSoA))),
        getTokenSoARecHitsExtra_(consumes(config.getParameter<edm::InputTag>(Traits::recHitsLayerClustersSoA))),
        detector_(config.getParameter<std::string>("detector")),
        timeClname_(config.getParameter<std::string>("timeClname")),
        algoId_(Traits::algoId(detector_)),
        caloID_(Traits::caloID(detector_)),
        producesMask_(Traits::producesMask(detector_)) {
#if DEBUG_CLUSTERS_ALPAKA
    moduleLabel_ = config.getParameter<std::string>("@module_label");
#endif
    if (producesMask_) {
      produces<std::vector<float>>("InitialLayerClustersMask");
    }
    produces<std::vector<reco::BasicCluster>>();
    produces<edm::ValueMap<std::pair<float, float>>>(timeClname_);
  }

  ~LayerClustersFromSoAProducerT() override = default;

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override {
    auto const& deviceData = iEvent.get(getTokenSoAClusters_);

    auto const& deviceSoARecHitsExtra = iEvent.get(getTokenSoARecHitsExtra_);
    auto const soaRecHitsExtra_v = deviceSoARecHitsExtra.view();

    auto const& deviceSoACells = iEvent.get(getTokenSoACells_);
    auto const soaCells_v = deviceSoACells.view();

    auto const deviceView = deviceData.view();

    std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>);
    clusters->reserve(deviceData->metadata().size()[0u]);

    for (auto i = 0; i < deviceData->metadata().size()[0u]; ++i) {
      std::vector<std::pair<DetId, float>> thisCluster;
      thisCluster.reserve(deviceView.position()[i].cells());
      clusters->emplace_back(
          deviceView.energy()[i].energy(),
          math::XYZPoint(deviceView.position()[i].x(), deviceView.position()[i].y(), deviceView.position()[i].z()),
          caloID_,
          std::move(thisCluster),
          algoId_);
      clusters->back().setSeed(deviceView.indexes()[i].seedID());
    }

    // Populate hits and fractions. This involves two SoAs: the original RecHits
    // SoA and the clustering algorithm's output SoA. Both SoAs have the same
    // cardinality, and crucially, the output SoA includes the cluster index.
    for (int32_t i = 0; i < soaRecHitsExtra_v.metadata().size(); ++i) {
      if (soaRecHitsExtra_v[i].clusterIndex() == -1) {
        continue;
      }
      assert(soaRecHitsExtra_v[i].clusterIndex() < (int)clusters->size());
      (*clusters)[soaRecHitsExtra_v[i].clusterIndex()].addHitAndFraction(soaCells_v[i].detid(), 1.f);
    }

    // The time of each cluster is computed on the device by the SoA layer clusters producer
    std::vector<std::pair<float, float>> cluster_times;
    cluster_times.reserve(clusters->size());
    for (unsigned i = 0; i < clusters->size(); ++i) {
      cluster_times.emplace_back(deviceView.timing()[i].time(), deviceView.timing()[i].timeError());
    }

#if DEBUG_CLUSTERS_ALPAKA
    auto runNumber = iEvent.eventAuxiliary().run();
    auto lumiNumber = iEvent.eventAuxiliary().luminosityBlock();
    auto evtNumber = iEvent.eventAuxiliary().id().event();

    hgcalUtils::DumpCellsSoA dumperCellsSoA;
    dumperCellsSoA.dumpInfos(deviceSoACells, moduleLabel_, runNumber, lumiNumber, evtNumber);

    hgcalUtils::DumpClusters dumper;
    dumper.dumpInfos(*clusters, moduleLabel_, runNumber, lumiNumber, evtNumber, true);

    hgcalUtils::DumpClustersSoA dumperSoA;
    dumperSoA.dumpInfos(deviceSoARecHitsExtra, moduleLabel_, runNumber, lumiNumber, evtNumber);
#endif

    auto clusterHandle = iEvent.put(std::move(clusters));

    auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
    edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
    filler.insert(clusterHandle, cluster_times.begin(), cluster_times.end());
    filler.fill();
    iEvent.put(std::move(timeCl), timeClname_);

    if (producesMask_) {
      std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
      layerClustersMask->resize(clusterHandle->size(), 1.0);
      iEvent.put(std::move(layerClustersMask), "InitialLayerClustersMask");
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    Traits::fillDescriptions(desc);
    desc.add<std::string>("timeClname", "timeLayerCluster");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  edm::EDGetTokenT<reco::CaloClusterHostCollection> const getTokenSoAClusters_;
  edm::EDGetTokenT<typename Traits::RecHitsHostCollection> const getTokenSoACells_;
  edm::EDGetTokenT<typename Traits::RecHitsExtraHostCollection> const getTokenSoARecHitsExtra_;
  std::string detector_;
  std::string timeClname_;
  reco::CaloCluster::AlgoId algoId_;
  reco::CaloID::Detectors caloID_;
  bool producesMask_;
#if DEBUG_CLUSTERS_ALPAKA
  std::string moduleLabel_;
#endif
};

using HGCalLayerClustersFromSoAProducer = LayerClustersFromSoAProducerT<HGCalLayerClustersFromSoATraits>;
using BarrelLayerClustersFromSoAProducer = LayerClustersFromSoAProducerT<BarrelLayerClustersFromSoATraits>;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClustersFromSoAProducer);
DEFINE_FWK_MODULE(BarrelLayerClustersFromSoAProducer);
