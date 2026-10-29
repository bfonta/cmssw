// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <optional>
#include <string>
#include <utility>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

#include "HGCalTrackstersSoAAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Builds the tracksters (their layer clusters, energy, barycenter, time and principal components) from the layer
  // clusters and their clustering (the trackster of each layer cluster), of HGCAL or of the barrel
  class HGCalSoATrackstersProducer : public stream::SynchronizingEDProducer<> {
  public:
    HGCalSoATrackstersProducer(const edm::ParameterSet& config)
        : SynchronizingEDProducer(config),
          layerClustersToken_{consumes(config.getParameter<edm::InputTag>("hgcalLayerClustersSoA"))},
          layerClustersClusteringToken_{
              consumes(config.getParameter<edm::InputTag>("hgcalLayerClustersTrackstersSoA"))},
          trackstersToken_{produces()},
          barrel_{config.getParameter<std::string>("detector") == "Barrel"},
          pcaCleaning_{config.getParameter<bool>("usePCACleaning")} {
      // the HGCAL geometry gives the electromagnetic part of HGCAL
      if (not barrel_) {
        ticlGeomToken_ = consumesCollector().esConsumes<TICLGeomHost, CaloGeometryRecord>();
        ticlGeomLookupToken_ = consumesCollector().esConsumes<TICLGeomLookupHost, CaloGeometryRecord>();
        ticlGeomLayersToken_ = consumesCollector().esConsumes<TICLGeomLayersHost, CaloGeometryRecord>();
      }
    }

    // The numbers of tracksters and of their layer clusters are on the device, and the host needs them to allocate
    // the tracksters
    void acquire(const device::Event& iEvent, const device::EventSetup&) override {
      const auto layerClustersClustering = iEvent.get(layerClustersClusteringToken_).view();
      const auto device = alpaka::getDev(iEvent.queue());
      counts_ = cms::alpakatools::make_host_buffer<unsigned int[]>(iEvent.queue(), 2);
      alpaka::memcpy(iEvent.queue(),
                     cms::alpakatools::make_host_view<unsigned int>((*counts_)[0]),
                     cms::alpakatools::make_device_view<const unsigned int>(
                         device, layerClustersClustering.numberOfTrackstersScalar()));
      alpaka::memcpy(iEvent.queue(),
                     cms::alpakatools::make_host_view<unsigned int>((*counts_)[1]),
                     cms::alpakatools::make_device_view<const unsigned int>(
                         device, layerClustersClustering.numberOfVerticesScalar()));
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      HGCalTrackstersSoAAlgoWrapper::Parameters parameters{
          .barrel = barrel_, .zLimitEm = 0.f, .pcaCleaning = pcaCleaning_};
      if (not barrel_) {
        // The layer clusters up to the last CE-E layer count to the electromagnetic energy
        ticlgeom::Tools geometry;
        geometry.setGeometry(iSetup.getData(ticlGeomToken_),
                             iSetup.getData(ticlGeomLookupToken_),
                             iSetup.getData(ticlGeomLayersToken_));
        parameters.zLimitEm = geometry.getPositionLayer(geometry.lastLayerEE()).z();
      }

      const unsigned int nTracksters = (*counts_)[0];
      edm::LogPrint("HGCalSoATrackstersProducer") << nTracksters << " tracksters, " << (*counts_)[1]
                                                  << " layer clusters in them";
      ticl::TracksterDeviceCollection tracksters(
          iEvent.queue(), nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, (*counts_)[1]);
      algo_.run(iEvent.queue(),
                parameters,
                iEvent.get(layerClustersToken_).const_view(),
                iEvent.get(layerClustersClusteringToken_).const_view(),
                tracksters.view());
      iEvent.emplace(trackstersToken_, std::move(tracksters));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("hgcalLayerClustersSoA", edm::InputTag("hgcalSoAMergeLayerClusters"));
      desc.add<edm::InputTag>("hgcalLayerClustersTrackstersSoA", edm::InputTag("ticlCLUEstering"));
      desc.ifValue(edm::ParameterDescription<std::string>("detector", "HGCAL", true),
                   edm::allowedValues<std::string>("HGCAL", "Barrel"))
          ->setComment(
              "The electromagnetic energy is that of the layer clusters of CE-E in HGCAL, of ECAL in the barrel.");
      desc.add<bool>("usePCACleaning", true)
          ->setComment(
              "Whether the PCA only uses the most energetic layer cluster of each layer within 10 layers of the most "
              "energetic one.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::EDGetToken<reco::CaloClusterDeviceCollection> layerClustersToken_;
    const device::EDGetToken<HGCalSoALayerClustersExtraDeviceCollection> layerClustersClusteringToken_;
    const device::EDPutToken<ticl::TracksterDeviceCollection> trackstersToken_;
    const bool barrel_;
    const bool pcaCleaning_;
    // consumed for HGCAL only
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
    edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
    edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
    const HGCalTrackstersSoAAlgoWrapper algo_{};
    // The numbers of tracksters and of their layer clusters, copied from the device in acquire and used in produce.
    // Pinned, so that the copies do not block acquire.
    std::optional<cms::alpakatools::host_buffer<unsigned int[]>> counts_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSoATrackstersProducer);
