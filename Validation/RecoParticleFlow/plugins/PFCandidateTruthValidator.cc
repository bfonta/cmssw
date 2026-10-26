// Particle-flow candidate validation against the MC-truth graph.
//
// The reference is the pfReconstructable level of the graph (SimDataFormats/TruthInfo
// LevelFlag::PFReconstructable): generator-stable particles with at least two tracker or
// one calorimeter sim hit over their subgraph, split into the particle-flow types by pdg
// id and charge.
//
// There is no PF associator. A candidate reaches the truth through the associators of its
// constituents, read here from the event (SimGeneral/TruthGraphAssociatorProducers):
//   - its track, through the track associator, only if the candidate has a track;
//   - its calorimeter deposits, only if it has some: the PFClusters of its PF block
//     elements for a barrel candidate, the tracksters of its TICLCandidate for a pfTICL
//     candidate, which is built one to one and in order from ticlCandidate.
// The track and calorimeter scores stay separate (PFTruthScores.h). Each configured
// ScoreMixer turns them into one score per direction and PFMatchDecider into a match
// (PFScoreMixer.h). Every mixer carries its own thresholds as equal-length vectors, and
// every (mixer, threshold point) gets its own folder of metrics.
//
// The metrics, per particle-flow type and as numerator/denominator pairs that
// DQMGenericClient divides at harvesting:
//   truth-driven (SimToReco)  efficiency, cumulatively: associated, then also of the right
//                             type, then also of the right energy; duplicate rate
//   reco-driven  (RecoToSim)  fake rate, cumulatively split into no match, wrong type and
//                             wrong energy; merge rate
// plus the candidate kinematics, the raw scores, the type confusion and the energy
// response. Each candidate source has its own folder, and a combined one pools them, so
// the barrel and the endcap are measured apart and together.

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "Validation/RecoParticleFlow/interface/PFScoreMixer.h"
#include "Validation/RecoParticleFlow/interface/PFTruthScores.h"

namespace {

  using AssociationMap = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;

  // The four axes every metric is binned in.
  enum Axis { kPt = 0, kEta, kPhi, kEnergy, kNAxes };
  constexpr std::array<const char*, kNAxes> kAxisNames = {{"pt", "eta", "phi", "energy"}};
  constexpr std::array<const char*, kNAxes> kAxisTitles = {{"p_{T} [GeV]", "#eta", "#phi", "E [GeV]"}};
  constexpr std::array<int, kNAxes> kAxisBins = {{100, 100, 64, 100}};
  constexpr std::array<double, kNAxes> kAxisMin = {{0., -5., -M_PI, 0.}};
  constexpr std::array<double, kNAxes> kAxisMax = {{200., 5., M_PI, 500.}};

  struct Kinematics {
    double pt = 0., eta = 0., phi = 0., energy = 0.;
    [[nodiscard]] std::array<double, kNAxes> axes() const { return {{pt, eta, phi, energy}}; }
  };

  // How a candidate source reaches its calorimeter constituents.
  enum class CaloLink { Blocks, TICLCandidates };

  // One entry of a reco object's score list: a reference truth particle, the score in one
  // direction, and the shared quantity of that direction (shared hits or energy reco-side,
  // the sim-normalised shared fraction truth-side).
  struct Entry {
    uint32_t truthId;
    float score;
    float shared;
  };

  // The reference-filtered rows of one associated collection, per reco object, in both
  // directions. TruthToReco is inverted once per event so a candidate reads its rows by
  // its own index.
  struct ScoreTable {
    std::vector<std::vector<Entry>> recoToSim;
    std::vector<std::vector<Entry>> simToReco;

    void fill(AssociationMap const& recoToTruth,
              AssociationMap const& truthToReco,
              std::size_t nReco,
              std::vector<uint8_t> const& isReference) {
      recoToSim.assign(nReco, {});
      simToReco.assign(nReco, {});
      for (std::size_t k = 0; k < std::min<std::size_t>(nReco, recoToTruth.size()); ++k) {
        for (auto const& element : recoToTruth[k]) {
          if (element.index() < isReference.size() && isReference[element.index()] != 0)
            recoToSim[k].push_back({element.index(), element.score(), element.sharedEnergy()});
        }
      }
      for (std::size_t t = 0; t < std::min<std::size_t>(isReference.size(), truthToReco.size()); ++t) {
        if (isReference[t] == 0)
          continue;
        for (auto const& element : truthToReco[t]) {
          if (element.index() < nReco)
            simToReco[element.index()].push_back({static_cast<uint32_t>(t), element.score(), element.sharedEnergy()});
        }
      }
    }
  };

  [[nodiscard]] Entry const* findEntry(std::vector<Entry> const& entries, uint32_t truthId) {
    for (auto const& entry : entries)
      if (entry.truthId == truthId)
        return &entry;
    return nullptr;
  }

  // A calorimetric collection with its association maps: the PFClusters of one
  // subdetector, or a trackster collection. Configuration only; what an event holds for
  // it is a CaloGroupTables.
  struct CaloGroup {
    std::string label;
    bool isTrackster = false;
    edm::EDGetTokenT<std::vector<reco::PFCluster>> clusterToken;
    edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken;
    edm::EDGetTokenT<AssociationMap> recoToTruthToken;
    edm::EDGetTokenT<AssociationMap> truthToRecoToken;
  };

  // One event's view of a calorimetric collection: which product it is, so a candidate's
  // constituent can be recognised by its ProductID, the energy that weights each object,
  // and the reference-filtered association rows.
  struct CaloGroupTables {
    edm::ProductID productId;
    std::vector<float> energies;
    ScoreTable table;
  };

  // Everything one event's candidates are scored from, built once in analyze and passed
  // down, so no per-event state lives in the module.
  struct EventTables {
    edm::ProductID trackProductId;
    ScoreTable track;
    std::vector<CaloGroupTables> calo;  // parallel to the configured caloGroups_

    [[nodiscard]] std::optional<std::size_t> caloGroupOf(edm::ProductID const& id) const {
      for (std::size_t g = 0; g < calo.size(); ++g)
        if (calo[g].productId == id)
          return g;
      return std::nullopt;
    }
  };

  // One calorimetric constituent of a candidate.
  struct CaloConstituent {
    std::size_t group;
    std::size_t index;
  };

  struct CandidateSource {
    std::string name;
    CaloLink caloLink = CaloLink::Blocks;
    edm::EDGetTokenT<std::vector<reco::PFCandidate>> candidateToken;
    edm::EDGetTokenT<std::vector<TICLCandidate>> ticlCandidateToken;
  };

  // A candidate as the metrics see it. Only typed candidates get one.
  struct CandidateRecord {
    Kinematics kinematics;
    pftruth::PFType type = pftruth::PFType::ChargedHadron;
    int pdgId = 0;
    int charge = 0;
    bool passesSelection = false;
    unsigned nCaloConstituents = 0;
    double ecalEnergy = 0., hcalEnergy = 0.;
    pftruth::CandidateScores scores;
  };

  // A reference truth particle as the metrics see it.
  struct TruthRecord {
    Kinematics kinematics;
    pftruth::PFType type = pftruth::PFType::ChargedHadron;
    bool passesSelection = false;
  };

  using ME = dqm::reco::MonitorElement*;
  using AxisRow = std::array<ME, kNAxes>;

  // The metric numerators and denominators of one (source, mixer, particle-flow type).
  struct MetricHistograms {
    // Truth-driven.
    AxisRow den{}, numAssoc{}, numPid{}, numEnergy{}, numDuplicate{}, notTouched{};
    // Reco-driven.
    AxisRow denFake{}, numFakeAssoc{}, numFakePid{}, numFakeEnergy{}, numFakeTotal{}, numMerge{};
    ME bestSimToRecoScore = nullptr, bestRecoToSimScore = nullptr;
    ME response = nullptr, responseVsEta = nullptr, nMatchedTruth = nullptr, nMatchedCandidates = nullptr;
  };

  // The metrics of one decision point, a (mixer, thresholds) pair.
  struct DecisionHistograms {
    std::array<MetricHistograms, pftruth::kNPFTypes> perType;
    // Truth type of the best match against candidate type, with an extra "none" bin for
    // a candidate matched to nothing.
    ME recoConfusion = nullptr;
    // Truth type against the type of the best candidate, with an extra "none" bin for a
    // particle no candidate matches.
    ME truthConfusion = nullptr;
  };

  // Mixer-independent plots of one (source, candidate type).
  struct CandidateHistograms {
    AxisRow kinematics{};
    ME charge = nullptr, pdgId = nullptr, hasTrack = nullptr, nCaloConstituents = nullptr;
    ME ecalEnergy = nullptr, hcalEnergy = nullptr, nTruthTouched = nullptr;
    // The best score of each kind over the truth particles the candidate touches.
    ME trackRecoToSim = nullptr, trackSimToReco = nullptr, caloRecoToSim = nullptr, caloSimToReco = nullptr;
    // Track against calorimeter score of the pair with the best track score.
    ME trackVsCaloRecoToSim = nullptr, trackVsCaloSimToReco = nullptr;
  };

  struct SourceHistograms {
    std::array<CandidateHistograms, pftruth::kNPFTypes> candidates;
    std::vector<DecisionHistograms> points;  // parallel to points_
  };

  struct TruthHistograms {
    AxisRow kinematics{};
    ME nTrackerSimHits = nullptr, nCaloSimHits = nullptr;
  };

}  // namespace

class PFCandidateTruthValidator : public DQMEDAnalyzer {
public:
  explicit PFCandidateTruthValidator(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  // One decision point: which mixer produces its mixed scores, and its thresholds.
  struct DecisionPoint {
    std::size_t mixer;
    pftruth::PFMatchDecider decider;
  };

  void bookSource(DQMStore::IBooker&, std::string const& name, SourceHistograms&);
  [[nodiscard]] pftruth::CandidateScores buildScores(EventTables const&,
                                                     std::optional<std::size_t> trackIndex,
                                                     std::vector<CaloConstituent> const& calo) const;
  [[nodiscard]] static std::optional<std::size_t> trackIndexOf(reco::PFCandidate const&,
                                                               edm::ProductID const& trackProductId);
  void fillCandidatePlots(SourceHistograms&, std::vector<CandidateRecord const*> const& candidates) const;
  void fillSource(SourceHistograms&,
                  std::vector<CandidateRecord const*> const& candidates,
                  std::vector<TruthRecord> const& truth,
                  std::vector<uint32_t> const& selectedReference) const;

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<AssociationMap> trackRecoToTruthToken_;
  const edm::EDGetTokenT<AssociationMap> trackTruthToRecoToken_;
  std::vector<CaloGroup> caloGroups_;
  std::vector<CandidateSource> sources_;
  std::vector<std::unique_ptr<pftruth::ScoreMixer>> mixers_;
  // Grouped by mixer, in configuration order: the points of one mixer are contiguous.
  std::vector<DecisionPoint> points_;
  const std::string dirName_;
  const double truthPtMin_, truthAbsEtaMax_;
  const bool truthSignalOnly_;
  const double recoPtMin_, recoAbsEtaMax_;

  bool warnedNoReference_ = false;

  std::vector<SourceHistograms> sourceHistograms_;  // one per source, then the combined one
  std::array<TruthHistograms, pftruth::kNPFTypes> truthHistograms_;
};

PFCandidateTruthValidator::PFCandidateTruthValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes(cfg.getParameter<edm::InputTag>("graph"))),
      trackToken_(consumes(cfg.getParameter<edm::InputTag>("tracks"))),
      trackRecoToTruthToken_(consumes(edm::InputTag(cfg.getParameter<std::string>("trackAssociator"),
                                                    cfg.getParameter<edm::InputTag>("tracks").label() + "RecoToTruth" +
                                                        cfg.getParameter<std::string>("workingPoint")))),
      trackTruthToRecoToken_(
          consumes(edm::InputTag(cfg.getParameter<std::string>("trackAssociator"),
                                 cfg.getParameter<edm::InputTag>("tracks").label() + "TruthToReco"))),
      dirName_(cfg.getParameter<std::string>("dirName")),
      truthPtMin_(cfg.getParameter<double>("truthPtMin")),
      truthAbsEtaMax_(cfg.getParameter<double>("truthAbsEtaMax")),
      truthSignalOnly_(cfg.getParameter<bool>("truthSignalOnly")),
      recoPtMin_(cfg.getParameter<double>("recoPtMin")),
      recoAbsEtaMax_(cfg.getParameter<double>("recoAbsEtaMax")) {
  const auto workingPoint = cfg.getParameter<std::string>("workingPoint");
  for (auto const& pset : cfg.getParameter<std::vector<edm::ParameterSet>>("caloAssociations")) {
    CaloGroup group;
    const auto collection = pset.getParameter<edm::InputTag>("collection");
    const auto associator = pset.getParameter<std::string>("associator");
    // The instance key rule of truthGraphAssociationLabels_cff.instanceKey.
    std::string key = collection.label();
    if (!collection.instance().empty())
      key += "_" + collection.instance();
    group.label = key;
    const auto type = pset.getParameter<std::string>("type");
    if (type == "PFCluster") {
      group.clusterToken = consumes(collection);
    } else if (type == "Trackster") {
      group.isTrackster = true;
      group.tracksterToken = consumes(collection);
    } else {
      throw cms::Exception("Configuration")
          << "caloAssociations type must be PFCluster or Trackster, got '" << type << "'";
    }
    group.recoToTruthToken = consumes(edm::InputTag(associator, key + "RecoToTruth" + workingPoint));
    group.truthToRecoToken = consumes(edm::InputTag(associator, key + "TruthToReco"));
    caloGroups_.push_back(std::move(group));
  }

  for (auto const& pset : cfg.getParameter<std::vector<edm::ParameterSet>>("candidateSources")) {
    CandidateSource source;
    source.name = pset.getParameter<std::string>("name");
    source.candidateToken = consumes(pset.getParameter<edm::InputTag>("candidates"));
    const auto link = pset.getParameter<std::string>("caloLink");
    if (link == "blocks") {
      source.caloLink = CaloLink::Blocks;
    } else if (link == "ticlCandidates") {
      source.caloLink = CaloLink::TICLCandidates;
      source.ticlCandidateToken = consumes(pset.getParameter<edm::InputTag>("ticlCandidates"));
    } else {
      throw cms::Exception("Configuration") << "caloLink must be blocks or ticlCandidates, got '" << link << "'";
    }
    sources_.push_back(std::move(source));
  }
  if (sources_.empty())
    throw cms::Exception("Configuration") << "candidateSources is empty";

  // One decider per (mixer, decision point): the mixing type and every threshold come
  // from the mixer PSet, and a vector of thresholds is a scan, one folder per point.
  for (auto const& pset : cfg.getParameter<std::vector<edm::ParameterSet>>("mixers")) {
    mixers_.push_back(pftruth::makeScoreMixer(pset));
    for (auto& thresholds : pftruth::decisionThresholds(pset))
      points_.push_back({mixers_.size() - 1, pftruth::PFMatchDecider(std::move(thresholds))});
  }
  if (mixers_.empty())
    throw cms::Exception("Configuration") << "mixers is empty: every metric is booked per mixer";
  for (std::size_t i = 0; i < mixers_.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (mixers_[i]->name() == mixers_[j]->name())
        throw cms::Exception("Configuration")
            << "two mixers are named '" << mixers_[i]->name() << "', and the name is their folder";
}

void PFCandidateTruthValidator::bookSource(DQMStore::IBooker& ibook, std::string const& name, SourceHistograms& h) {
  using pftruth::kPFTypeNames;
  const std::string base = dirName_ + "/" + name;
  for (std::size_t type = 0; type < pftruth::kNPFTypes; ++type) {
    const std::string typeName = kPFTypeNames[type];
    auto& c = h.candidates[type];
    ibook.setCurrentFolder(base + "/Candidates/" + typeName);
    for (int a = 0; a < kNAxes; ++a)
      c.kinematics[a] = ibook.book1D(std::string("cand_") + kAxisNames[a],
                                     typeName + " candidates;" + kAxisTitles[a] + ";candidates",
                                     kAxisBins[a],
                                     kAxisMin[a],
                                     kAxisMax[a]);
    c.charge = ibook.book1D("cand_charge", typeName + " candidates;charge;candidates", 3, -1.5, 1.5);
    c.pdgId = ibook.book1D("cand_pdgId", typeName + " candidates;PDG id;candidates", 441, -220.5, 220.5);
    c.hasTrack =
        ibook.book1D("cand_hasTrack", typeName + " candidates;has an associated track;candidates", 2, -0.5, 1.5);
    c.nCaloConstituents = ibook.book1D("cand_nCaloConstituents",
                                       typeName + " candidates;associated calorimeter constituents;candidates",
                                       21,
                                       -0.5,
                                       20.5);
    c.ecalEnergy =
        ibook.book1D("cand_ecalEnergy", typeName + " candidates;ECAL energy [GeV];candidates", 100, 0., 500.);
    c.hcalEnergy =
        ibook.book1D("cand_hcalEnergy", typeName + " candidates;HCAL energy [GeV];candidates", 100, 0., 500.);
    c.nTruthTouched = ibook.book1D(
        "cand_nTruthTouched", typeName + " candidates;reference particles touched;candidates", 21, -0.5, 20.5);

    ibook.setCurrentFolder(base + "/Scores/" + typeName);
    auto score = [&](const char* name, const char* title) {
      return ibook.book1D(name, typeName + " candidates;" + title + ";candidates", 50, 0., 1.);
    };
    c.trackRecoToSim = score("score_track_recoToSim", "best track RecoToSim score");
    c.trackSimToReco = score("score_track_simToReco", "best track SimToReco score");
    c.caloRecoToSim = score("score_calo_recoToSim", "best calorimeter RecoToSim score");
    c.caloSimToReco = score("score_calo_simToReco", "best calorimeter SimToReco score");
    c.trackVsCaloRecoToSim = ibook.book2D("score_trackVsCalo_recoToSim",
                                          typeName +
                                              " candidates, pair of the best track score;track RecoToSim "
                                              "score;calorimeter RecoToSim score",
                                          50,
                                          0.,
                                          1.,
                                          50,
                                          0.,
                                          1.);
    c.trackVsCaloSimToReco = ibook.book2D("score_trackVsCalo_simToReco",
                                          typeName +
                                              " candidates, pair of the best track score;track SimToReco "
                                              "score;calorimeter SimToReco score",
                                          50,
                                          0.,
                                          1.,
                                          50,
                                          0.,
                                          1.);
  }

  h.points.resize(points_.size());
  for (std::size_t m = 0; m < points_.size(); ++m) {
    auto& mh = h.points[m];
    const std::string mixerBase =
        base + "/Metrics/" + mixers_[points_[m].mixer]->name() + "/" + points_[m].decider.thresholds().label;
    ibook.setCurrentFolder(mixerBase);
    mh.recoConfusion = ibook.book2D("confusion_reco",
                                    "Candidate type against the type of its best match;best truth "
                                    "match;candidate",
                                    pftruth::kNPFTypes + 1,
                                    -0.5,
                                    pftruth::kNPFTypes + 0.5,
                                    pftruth::kNPFTypes,
                                    -0.5,
                                    pftruth::kNPFTypes - 0.5);
    mh.truthConfusion = ibook.book2D("confusion_truth",
                                     "Truth type against the type of its best candidate;truth;best candidate",
                                     pftruth::kNPFTypes,
                                     -0.5,
                                     pftruth::kNPFTypes - 0.5,
                                     pftruth::kNPFTypes + 1,
                                     -0.5,
                                     pftruth::kNPFTypes + 0.5);
    for (std::size_t type = 0; type < pftruth::kNPFTypes; ++type) {
      mh.recoConfusion->setBinLabel(type + 1, kPFTypeNames[type], 1);
      mh.recoConfusion->setBinLabel(type + 1, kPFTypeNames[type], 2);
      mh.truthConfusion->setBinLabel(type + 1, kPFTypeNames[type], 1);
      mh.truthConfusion->setBinLabel(type + 1, kPFTypeNames[type], 2);
    }
    mh.recoConfusion->setBinLabel(pftruth::kNPFTypes + 1, "none", 1);
    mh.truthConfusion->setBinLabel(pftruth::kNPFTypes + 1, "none", 2);

    for (std::size_t type = 0; type < pftruth::kNPFTypes; ++type) {
      const std::string typeName = kPFTypeNames[type];
      auto& t = mh.perType[type];
      ibook.setCurrentFolder(mixerBase + "/" + typeName);
      auto row = [&](AxisRow& out, const char* prefix, std::string const& what) {
        for (int a = 0; a < kNAxes; ++a)
          out[a] = ibook.book1D(std::string(prefix) + "_" + kAxisNames[a],
                                typeName + " " + what + ";" + kAxisTitles[a] + ";entries",
                                kAxisBins[a],
                                kAxisMin[a],
                                kAxisMax[a]);
      };
      row(t.den, "den", "reference particles");
      row(t.numAssoc, "num_assoc", "reference particles matched to a candidate");
      row(t.numPid, "num_pid", "reference particles matched to a candidate of their type");
      row(t.numEnergy, "num_energy", "reference particles matched to a candidate of their type and energy");
      row(t.numDuplicate, "num_duplicate", "reference particles matched to more than one candidate");
      row(t.notTouched, "num_notTouched", "reference particles no candidate constituent touches");
      row(t.denFake, "den_fake", "candidates");
      row(t.numFakeAssoc, "num_fake_assoc", "candidates matched to no reference particle");
      row(t.numFakePid, "num_fake_pid", "candidates matched to a reference particle of another type");
      row(t.numFakeEnergy, "num_fake_energy", "candidates matched to a reference particle of their type, wrong energy");
      row(t.numFakeTotal, "num_fake_total", "fake candidates, any reason");
      row(t.numMerge, "num_merge", "candidates matched to more than one reference particle");
      t.bestSimToRecoScore = ibook.book1D(
          "score_best_simToReco", typeName + " reference particles;best mixed SimToReco score;entries", 50, 0., 1.);
      t.bestRecoToSimScore =
          ibook.book1D("score_best_recoToSim", typeName + " candidates;best mixed RecoToSim score;entries", 50, 0., 1.);
      t.response = ibook.book1D(
          "response", typeName + " reference particles, best candidate;E_{reco}/E_{true};entries", 100, 0., 3.);
      t.responseVsEta = ibook.book2D("response_vs_eta",
                                     typeName + " reference particles, best candidate;#eta;E_{reco}/E_{true}",
                                     50,
                                     -5.,
                                     5.,
                                     60,
                                     0.,
                                     3.);
      t.nMatchedTruth = ibook.book1D(
          "nMatchedTruth", typeName + " candidates;matched reference particles;candidates", 11, -0.5, 10.5);
      t.nMatchedCandidates = ibook.book1D(
          "nMatchedCandidates", typeName + " reference particles;matched candidates;particles", 11, -0.5, 10.5);
    }
  }
}

void PFCandidateTruthValidator::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const&) {
  sourceHistograms_.assign(sources_.size() + (sources_.size() > 1 ? 1 : 0), {});
  for (std::size_t s = 0; s < sources_.size(); ++s)
    bookSource(ibook, sources_[s].name, sourceHistograms_[s]);
  if (sources_.size() > 1)
    bookSource(ibook, "combined", sourceHistograms_.back());

  for (std::size_t type = 0; type < pftruth::kNPFTypes; ++type) {
    const std::string typeName = pftruth::kPFTypeNames[type];
    auto& t = truthHistograms_[type];
    ibook.setCurrentFolder(dirName_ + "/Truth/" + typeName);
    for (int a = 0; a < kNAxes; ++a)
      t.kinematics[a] = ibook.book1D(std::string("truth_") + kAxisNames[a],
                                     typeName + " reference particles;" + kAxisTitles[a] + ";particles",
                                     kAxisBins[a],
                                     kAxisMin[a],
                                     kAxisMax[a]);
    t.nTrackerSimHits = ibook.book1D(
        "truth_nTrackerSimHits", typeName + " reference particles;own tracker sim hits;particles", 51, -0.5, 50.5);
    t.nCaloSimHits = ibook.book1D(
        "truth_nCaloSimHits", typeName + " reference particles;own calorimeter sim hits;particles", 101, -0.5, 1000.5);
  }
}

std::optional<std::size_t> PFCandidateTruthValidator::trackIndexOf(reco::PFCandidate const& candidate,
                                                                   edm::ProductID const& trackProductId) {
  auto const& track = candidate.trackRef();
  if (track.isNonnull() && track.id() == trackProductId)
    return track.key();
  // A PF muon carries the refitted track its momentum comes from, which is not a
  // generalTracks entry; its inner track is.
  auto const& muon = candidate.muonRef();
  if (muon.isNonnull() && muon.isAvailable()) {
    auto const& inner = muon->innerTrack();
    if (inner.isNonnull() && inner.id() == trackProductId)
      return inner.key();
  }
  return std::nullopt;
}

pftruth::CandidateScores PFCandidateTruthValidator::buildScores(EventTables const& tables,
                                                                std::optional<std::size_t> trackIndex,
                                                                std::vector<CaloConstituent> const& calo) const {
  pftruth::CandidateScores scores;
  std::unordered_map<uint32_t, std::size_t> pairOf;
  auto pairFor = [&](uint32_t truthId) -> pftruth::PairScores& {
    auto [it, inserted] = pairOf.try_emplace(truthId, scores.pairs.size());
    if (inserted) {
      scores.pairs.emplace_back();
      scores.pairs.back().truthId = truthId;
    }
    return scores.pairs[it->second];
  };

  // Track: one constituent, its scores taken as they are.
  if (trackIndex && *trackIndex < tables.track.recoToSim.size()) {
    scores.hasTrack = true;
    for (auto const& entry : tables.track.recoToSim[*trackIndex])
      pairFor(entry.truthId).trackRecoToSim = entry.score;
    for (auto const& entry : tables.track.simToReco[*trackIndex])
      pairFor(entry.truthId).trackSimToReco = entry.score;
  }

  // Calorimeter: several constituents, combined per truth particle.
  //   RecoToSim  the energy-weighted mean of the constituents' reco-normalised scores, a
  //              constituent that does not touch the particle counting as 1: how much of
  //              the candidate's calorimetric energy the particle fails to explain.
  //   SimToReco  one minus the coverage, the energy the candidate's constituents share
  //              with the particle over the particle's own energy in the calorimetric
  //              groups the candidate has deposits in. Inside one group the sim-normalised
  //              fractions have one denominator and add; across groups the denominators
  //              are recovered as shared / fraction, so ECAL and HCAL add by energy.
  double totalWeight = 0.;
  for (auto const& c : calo)
    totalWeight += std::max(0.f, tables.calo[c.group].energies[c.index]);
  if (!calo.empty()) {
    scores.hasCalo = true;
    const std::size_t nGroups = tables.calo.size();
    struct Accumulator {
      double explained = 0.;  // the energy-weighted sum of 1 - score
      double shared = 0.;
      std::vector<double> groupFraction, groupTruthEnergy;
    };
    std::unordered_map<uint32_t, Accumulator> accumulators;
    auto accumulatorFor = [&](uint32_t truthId) -> Accumulator& {
      auto [it, inserted] = accumulators.try_emplace(truthId);
      if (inserted) {
        it->second.groupFraction.assign(nGroups, 0.);
        it->second.groupTruthEnergy.assign(nGroups, 0.);
      }
      return it->second;
    };
    for (auto const& c : calo) {
      auto const& group = tables.calo[c.group];
      // Zero-energy constituents everywhere would leave no weight; count them equally.
      const double weight = totalWeight > 0. ? std::max(0.f, group.energies[c.index]) / totalWeight : 1. / calo.size();
      auto const& recoToSim = group.table.recoToSim[c.index];
      for (auto const& entry : recoToSim) {
        auto& acc = accumulatorFor(entry.truthId);
        acc.explained += weight * (1. - entry.score);
        acc.shared += entry.shared;
      }
      for (auto const& entry : group.table.simToReco[c.index]) {
        auto& acc = accumulatorFor(entry.truthId);
        acc.groupFraction[c.group] += entry.shared;
        if (acc.groupTruthEnergy[c.group] <= 0. && entry.shared > 0.f) {
          if (auto const* reco = findEntry(recoToSim, entry.truthId); reco != nullptr && reco->shared > 0.f)
            acc.groupTruthEnergy[c.group] = reco->shared / entry.shared;
        }
      }
    }
    for (auto const& [truthId, acc] : accumulators) {
      auto& pair = pairFor(truthId);
      // The untouched constituents explain nothing, so they add 0 to explained.
      pair.caloRecoToSim = static_cast<float>(std::clamp(1. - acc.explained, 0., 1.));
      double sharedByEnergy = 0., truthEnergy = 0., bestFraction = 0.;
      for (std::size_t g = 0; g < nGroups; ++g) {
        bestFraction = std::max(bestFraction, acc.groupFraction[g]);
        if (acc.groupTruthEnergy[g] > 0.) {
          sharedByEnergy += acc.groupFraction[g] * acc.groupTruthEnergy[g];
          truthEnergy += acc.groupTruthEnergy[g];
        }
      }
      // With no denominator recovered the best group's own fraction stands in.
      const double coverage = truthEnergy > 0. ? sharedByEnergy / truthEnergy : bestFraction;
      pair.caloSimToReco = static_cast<float>(1. - std::clamp(coverage, 0., 1.));
      pair.caloSharedEnergy = static_cast<float>(acc.shared);
      pair.caloTruthEnergy = static_cast<float>(truthEnergy);
    }
  }

  // A constituent the candidate has but that does not touch a particle says "no match"
  // for it, which is the worst score, not the absence of one.
  for (auto& pair : scores.pairs) {
    if (scores.hasTrack) {
      pair.trackRecoToSim = pair.trackRecoToSim.value_or(1.f);
      pair.trackSimToReco = pair.trackSimToReco.value_or(1.f);
    }
    if (scores.hasCalo) {
      pair.caloRecoToSim = pair.caloRecoToSim.value_or(1.f);
      pair.caloSimToReco = pair.caloSimToReco.value_or(1.f);
    }
  }
  return scores;
}

void PFCandidateTruthValidator::fillCandidatePlots(SourceHistograms& h,
                                                   std::vector<CandidateRecord const*> const& candidates) const {
  using pftruth::Constituent;
  using pftruth::Direction;
  for (auto const* candidate : candidates) {
    if (!candidate->passesSelection)
      continue;
    auto& c = h.candidates[static_cast<std::size_t>(candidate->type)];
    const auto axes = candidate->kinematics.axes();
    for (int a = 0; a < kNAxes; ++a)
      c.kinematics[a]->Fill(axes[a]);
    c.charge->Fill(candidate->charge);
    c.pdgId->Fill(candidate->pdgId);
    c.hasTrack->Fill(candidate->scores.hasTrack ? 1 : 0);
    c.nCaloConstituents->Fill(candidate->nCaloConstituents);
    c.ecalEnergy->Fill(candidate->ecalEnergy);
    c.hcalEnergy->Fill(candidate->hcalEnergy);
    c.nTruthTouched->Fill(candidate->scores.pairs.size());
    auto best = [&](Constituent constituent, Direction direction) {
      std::optional<float> out;
      for (auto const& pair : candidate->scores.pairs)
        if (auto const s = pair.score(constituent, direction); s && (!out || *s < *out))
          out = s;
      return out;
    };
    if (auto s = best(Constituent::Track, Direction::RecoToSim))
      c.trackRecoToSim->Fill(*s);
    if (auto s = best(Constituent::Track, Direction::SimToReco))
      c.trackSimToReco->Fill(*s);
    if (auto s = best(Constituent::Calo, Direction::RecoToSim))
      c.caloRecoToSim->Fill(*s);
    if (auto s = best(Constituent::Calo, Direction::SimToReco))
      c.caloSimToReco->Fill(*s);
    if (candidate->scores.hasTrack && candidate->scores.hasCalo && !candidate->scores.pairs.empty()) {
      // Both kinds are filled for every pair of such a candidate, so value() is safe.
      auto const& pair = *std::min_element(
          candidate->scores.pairs.begin(), candidate->scores.pairs.end(), [](auto const& x, auto const& y) {
            return x.trackRecoToSim.value() < y.trackRecoToSim.value();
          });
      c.trackVsCaloRecoToSim->Fill(pair.trackRecoToSim.value(), pair.caloRecoToSim.value());
      c.trackVsCaloSimToReco->Fill(pair.trackSimToReco.value(), pair.caloSimToReco.value());
    }
  }
}

void PFCandidateTruthValidator::fillSource(SourceHistograms& h,
                                           std::vector<CandidateRecord const*> const& candidates,
                                           std::vector<TruthRecord> const& truth,
                                           std::vector<uint32_t> const& selectedReference) const {
  fillCandidatePlots(h, candidates);

  auto fillRow = [](AxisRow const& row, std::array<double, kNAxes> const& axes) {
    for (int a = 0; a < kNAxes; ++a)
      row[a]->Fill(axes[a]);
  };

  // The mixed pairs of every candidate, recomputed per mixer and shared by all of that
  // mixer's decision points, which are contiguous in points_.
  std::vector<std::vector<pftruth::MixedPair>> mixed(candidates.size());
  std::vector<pftruth::TruthDecision> perTruth(truth.size());
  std::optional<std::size_t> mixedBy;

  for (std::size_t m = 0; m < points_.size(); ++m) {
    auto const& point = points_[m];
    if (mixedBy != point.mixer) {
      for (std::size_t i = 0; i < candidates.size(); ++i)
        mixers_[point.mixer]->mixAll(candidates[i]->scores, mixed[i]);
      mixedBy = point.mixer;
    }
    auto const& decider = point.decider;
    auto& mh = h.points[m];
    const double maxDeviation = decider.thresholds().maxEnergyResponseDeviation;
    auto const energyIsRight = [maxDeviation](double reco, double true_) {
      return true_ > 0. && std::abs(reco / true_ - 1.) < maxDeviation;
    };

    // Reco-driven: fake and merge, over the candidates passing the reco selection, each
    // matched against every reference particle of any interaction.
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      auto const& candidate = *candidates[i];
      if (!candidate.passesSelection)
        continue;
      const auto type = static_cast<std::size_t>(candidate.type);
      auto& t = mh.perType[type];
      const auto axes = candidate.kinematics.axes();
      const auto decision = decider.decideReco(mixed[i]);
      fillRow(t.denFake, axes);
      t.nMatchedTruth->Fill(decision.nMatched);
      if (decision.nMatched > 1)
        fillRow(t.numMerge, axes);
      if (decision.bestTruth < 0) {
        fillRow(t.numFakeAssoc, axes);
        fillRow(t.numFakeTotal, axes);
        mh.recoConfusion->Fill(pftruth::kNPFTypes, type);
        continue;
      }
      t.bestRecoToSimScore->Fill(decision.bestScore);
      auto const& matched = truth[decision.bestTruth];
      mh.recoConfusion->Fill(static_cast<std::size_t>(matched.type), type);
      if (matched.type != candidate.type) {
        fillRow(t.numFakePid, axes);
        fillRow(t.numFakeTotal, axes);
      } else if (!energyIsRight(candidate.kinematics.energy, matched.kinematics.energy)) {
        fillRow(t.numFakeEnergy, axes);
        fillRow(t.numFakeTotal, axes);
      }
    }

    // Truth-driven: efficiency and duplicate, over the selected reference particles,
    // matched by any candidate whatever its reco selection.
    std::fill(perTruth.begin(), perTruth.end(), pftruth::TruthDecision{});
    for (std::size_t i = 0; i < candidates.size(); ++i)
      decider.accumulateTruth(mixed[i], static_cast<int64_t>(i), perTruth);
    for (const uint32_t id : selectedReference) {
      auto const& particle = truth[id];
      const auto type = static_cast<std::size_t>(particle.type);
      auto& t = mh.perType[type];
      const auto axes = particle.kinematics.axes();
      auto const& decision = perTruth[id];
      fillRow(t.den, axes);
      t.nMatchedCandidates->Fill(decision.nMatched);
      if (!decision.touched)
        fillRow(t.notTouched, axes);
      if (decision.nMatched > 1)
        fillRow(t.numDuplicate, axes);
      if (decision.bestCandidate < 0) {
        mh.truthConfusion->Fill(type, pftruth::kNPFTypes);
        continue;
      }
      fillRow(t.numAssoc, axes);
      t.bestSimToRecoScore->Fill(decision.bestScore);
      auto const& best = *candidates[decision.bestCandidate];
      if (particle.kinematics.energy > 0.) {
        const double response = best.kinematics.energy / particle.kinematics.energy;
        t.response->Fill(response);
        t.responseVsEta->Fill(particle.kinematics.eta, response);
      }
      mh.truthConfusion->Fill(type, static_cast<std::size_t>(best.type));
      if (best.type != particle.type)
        continue;
      fillRow(t.numPid, axes);
      if (energyIsRight(best.kinematics.energy, particle.kinematics.energy))
        fillRow(t.numEnergy, axes);
    }
  }
}

void PFCandidateTruthValidator::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  const uint32_t nParticles = graph.nParticles();

  // The reference: the pfReconstructable level, as stamped on the graph. isReference
  // filters the association rows; selectedReference is the efficiency denominator.
  std::vector<uint8_t> isReference(nParticles, 0);
  std::vector<TruthRecord> truth(nParticles);
  std::vector<uint32_t> selectedReference;
  for (uint32_t id = 0; id < nParticles; ++id) {
    auto const& data = graph.particles()[id];
    if (!data.isAtLevel(truth::LevelFlag::PFReconstructable))
      continue;
    isReference[id] = 1;
    auto& record = truth[id];
    auto const& p4 = data.momentum;
    record.kinematics = {p4.pt(), p4.eta(), p4.phi(), p4.energy()};
    record.type = pftruth::pfTypeFromTruth(data.pdgId, truth::Particle(&graph, id).charge());
    record.passesSelection =
        p4.pt() >= truthPtMin_ && std::abs(p4.eta()) <= truthAbsEtaMax_ && (!truthSignalOnly_ || data.isSignal());
    if (!record.passesSelection)
      continue;
    selectedReference.push_back(id);
    auto& th = truthHistograms_[static_cast<std::size_t>(record.type)];
    const auto axes = record.kinematics.axes();
    for (int a = 0; a < kNAxes; ++a)
      th.kinematics[a]->Fill(axes[a]);
    th.nTrackerSimHits->Fill(data.nTrackerSimHits);
    th.nCaloSimHits->Fill(data.nCaloSimHits);
  }
  // A graph written before the level existed carries no bit, and every plot would be
  // silently empty; a non-empty graph always has generator-stable particles with hits.
  if (nParticles > 0 && std::none_of(isReference.begin(), isReference.end(), [](uint8_t r) { return r != 0; }) &&
      !warnedNoReference_) {
    warnedNoReference_ = true;
    edm::LogWarning("PFCandidateTruthValidator")
        << "the truth graph has " << nParticles
        << " particles and none at the pfReconstructable level. A graph written before the level and the "
           "ParticleData sim-hit counts existed reads back with neither; rebuild it from the DIGI step. Reported "
           "once per job.";
  }

  // The association rows of every constituent collection, reference-filtered.
  EventTables tables;
  {
    auto const tracks = event.getHandle(trackToken_);
    tables.trackProductId = tracks.id();
    tables.track.fill(
        event.get(trackRecoToTruthToken_), event.get(trackTruthToRecoToken_), tracks->size(), isReference);
  }
  tables.calo.resize(caloGroups_.size());
  for (std::size_t g = 0; g < caloGroups_.size(); ++g) {
    auto const& group = caloGroups_[g];
    auto& out = tables.calo[g];
    if (group.isTrackster) {
      auto const handle = event.getHandle(group.tracksterToken);
      out.productId = handle.id();
      for (auto const& trackster : *handle)
        out.energies.push_back(trackster.raw_energy());
    } else {
      auto const handle = event.getHandle(group.clusterToken);
      out.productId = handle.id();
      for (auto const& cluster : *handle)
        out.energies.push_back(cluster.energy());
    }
    out.table.fill(
        event.get(group.recoToTruthToken), event.get(group.truthToRecoToken), out.energies.size(), isReference);
  }

  // One record per typed candidate of every source. A candidate with no particle-flow
  // type (PFCandidate::X) enters no metric on either side: it is neither a fake of some
  // type nor allowed to make a truth particle efficient.
  std::vector<std::vector<CandidateRecord>> records(sources_.size());
  for (std::size_t s = 0; s < sources_.size(); ++s) {
    auto const& source = sources_[s];
    auto const& candidates = event.get(source.candidateToken);
    std::vector<TICLCandidate> const* ticlCandidates = nullptr;
    if (source.caloLink == CaloLink::TICLCandidates) {
      ticlCandidates = &event.get(source.ticlCandidateToken);
      if (ticlCandidates->size() != candidates.size()) {
        edm::LogWarning("PFCandidateTruthValidator")
            << "source '" << source.name << "': " << candidates.size() << " candidates but " << ticlCandidates->size()
            << " TICLCandidates, so the one-to-one link does not hold and its candidates get no calorimeter score";
        ticlCandidates = nullptr;
      }
    }
    auto& out = records[s];
    out.reserve(candidates.size());
    std::vector<CaloConstituent> calo;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      auto const& candidate = candidates[i];
      const auto type = pftruth::pfTypeFromReco(candidate.pdgId());
      if (!type)
        continue;
      auto& record = out.emplace_back();
      record.kinematics = {candidate.pt(), candidate.eta(), candidate.phi(), candidate.energy()};
      record.pdgId = candidate.pdgId();
      record.charge = candidate.charge();
      record.type = *type;
      record.passesSelection = candidate.pt() >= recoPtMin_ && std::abs(candidate.eta()) <= recoAbsEtaMax_;
      record.ecalEnergy = candidate.ecalEnergy();
      record.hcalEnergy = candidate.hcalEnergy();

      // The constituents inside a configured calorimetric collection, each once.
      calo.clear();
      auto addConstituent = [&](edm::ProductID const& id, std::size_t key) {
        auto const g = tables.caloGroupOf(id);
        if (!g || key >= tables.calo[*g].energies.size())
          return;
        if (std::none_of(calo.begin(), calo.end(), [&](auto const& c) { return c.group == *g && c.index == key; }))
          calo.push_back({*g, key});
      };
      if (source.caloLink == CaloLink::Blocks) {
        for (auto const& [block, element] : candidate.elementsInBlocks()) {
          if (block.isNull() || !block.isAvailable() || element >= block->elements().size())
            continue;
          auto const& cluster = block->elements()[element].clusterRef();
          if (cluster.isNonnull())
            addConstituent(cluster.id(), cluster.key());
        }
      } else if (ticlCandidates != nullptr) {
        for (auto const& trackster : (*ticlCandidates)[i].tracksters())
          if (trackster.isNonnull())
            addConstituent(trackster.id(), trackster.key());
      }
      record.nCaloConstituents = calo.size();
      record.scores = buildScores(tables, trackIndexOf(candidate, tables.trackProductId), calo);
    }
  }

  std::vector<CandidateRecord const*> combined;
  for (std::size_t s = 0; s < sources_.size(); ++s) {
    std::vector<CandidateRecord const*> view;
    view.reserve(records[s].size());
    for (auto const& record : records[s])
      view.push_back(&record);
    fillSource(sourceHistograms_[s], view, truth, selectedReference);
    combined.insert(combined.end(), view.begin(), view.end());
  }
  if (sources_.size() > 1)
    fillSource(sourceHistograms_.back(), combined, truth, selectedReference);
}

void PFCandidateTruthValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("graph", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"))
      ->setComment("The track collection the track associator scored; its label is the association map key");
  desc.add<std::string>("trackAssociator", "allTrackToTruthBranchAssociators");
  desc.add<std::string>("workingPoint", "Fixed")
      ->setComment(
          "RecoToTruth working point. Fixed keeps every scored pair in each row, which a per-pair score needs; an "
          "adaptive point keeps one pair per reco object");

  edm::ParameterSetDescription calo;
  calo.add<edm::InputTag>("collection", edm::InputTag("particleFlowClusterECAL"));
  calo.add<std::string>("type", "PFCluster")->setComment("PFCluster or Trackster");
  calo.add<std::string>("associator", "truthBranchPFClusterEcalAssociators");
  desc.addVPSet("caloAssociations", calo, {[] {
                  edm::ParameterSet p;
                  p.addParameter<edm::InputTag>("collection", edm::InputTag("particleFlowClusterECAL"));
                  p.addParameter<std::string>("type", "PFCluster");
                  p.addParameter<std::string>("associator", "truthBranchPFClusterEcalAssociators");
                  return p;
                }()});

  edm::ParameterSetDescription source;
  source.add<std::string>("name", "barrel");
  source.add<edm::InputTag>("candidates", edm::InputTag("particleFlowTmpBarrel"));
  source.add<std::string>("caloLink", "blocks")
      ->setComment(
          "blocks: the PFClusters of the candidate's PF block elements. ticlCandidates: the tracksters of the "
          "TICLCandidate at the candidate's own index, the pfTICL convention");
  source.add<edm::InputTag>("ticlCandidates", edm::InputTag("ticlCandidate"));
  desc.addVPSet("candidateSources", source, {[] {
                  edm::ParameterSet p;
                  p.addParameter<std::string>("name", "barrel");
                  p.addParameter<edm::InputTag>("candidates", edm::InputTag("particleFlowTmpBarrel"));
                  p.addParameter<std::string>("caloLink", "blocks");
                  p.addParameter<edm::InputTag>("ticlCandidates", edm::InputTag("ticlCandidate"));
                  return p;
                }()});

  desc.addVPSet("mixers", pftruth::scoreMixerDescription(), {[] {
                  edm::ParameterSet p;
                  p.addParameter<std::string>("name", "TrackFirst");
                  p.addParameter<std::string>("type", "trackFirst");
                  p.addParameter<double>("trackWeight", 0.5);
                  p.addParameter<double>("caloWeight", 0.5);
                  p.addParameter<std::vector<double>>("maxSimToRecoScore", {0.5});
                  p.addParameter<std::vector<double>>("maxRecoToSimScore", {0.5});
                  p.addParameter<std::vector<double>>("maxEnergyResponseDeviation", {0.5});
                  return p;
                }()})
      ->setComment(
          "The mixings, each with its type, its parameters and its decision thresholds. The thresholds are "
          "vectors of one length, one entry per decision point; see scoreMixerDescription");
  desc.add<double>("truthPtMin", 1.);
  desc.add<double>("truthAbsEtaMax", 4.);
  desc.add<bool>("truthSignalOnly", true)
      ->setComment("Efficiency denominator from the signal interaction only. Fake and merge match any interaction");
  desc.add<double>("recoPtMin", 1.);
  desc.add<double>("recoAbsEtaMax", 4.);
  desc.add<std::string>("dirName", "ParticleFlow/PFCandidateTruth");
  descriptions.add("pfCandidateTruthValidator", desc);
}

DEFINE_FWK_MODULE(PFCandidateTruthValidator);
