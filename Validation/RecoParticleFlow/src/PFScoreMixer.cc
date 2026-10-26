#include "Validation/RecoParticleFlow/interface/PFScoreMixer.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <sstream>

#include "FWCore/Utilities/interface/Exception.h"

namespace pftruth {

  std::map<std::string, ScoreMixerFactory>& scoreMixerRegistry() {
    static std::map<std::string, ScoreMixerFactory> registry;
    return registry;
  }

  std::unique_ptr<ScoreMixer> makeScoreMixer(edm::ParameterSet const& cfg) {
    auto const type = cfg.getParameter<std::string>("type");
    auto const& registry = scoreMixerRegistry();
    auto const it = registry.find(type);
    if (it == registry.end()) {
      std::ostringstream known;
      for (auto const& [name, factory] : registry)
        known << " " << name;
      throw cms::Exception("Configuration") << "unknown score mixer type '" << type << "', known:" << known.str();
    }
    return it->second(cfg);
  }

  edm::ParameterSetDescription scoreMixerDescription() {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "TrackFirst")->setComment("Label of the mixing in the DQM folder names");
    desc.add<std::string>("type", "trackFirst")
        ->setComment(
            "trackFirst, caloFirst, trackOnly, caloOnly, weighted, best or worst; see "
            "Validation/RecoParticleFlow/src/PFScoreMixer.cc");
    desc.add<double>("trackWeight", 0.5)->setComment("weighted only: the weight of the track score");
    desc.add<double>("caloWeight", 0.5)->setComment("weighted only: the weight of the calorimeter score");
    desc.add<std::vector<double>>("maxSimToRecoScore", {0.5})
        ->setComment(
            "A truth particle is matched by a candidate whose mixed SimToReco score is below this; one entry per "
            "decision point");
    desc.add<std::vector<double>>("maxRecoToSimScore", {0.5})
        ->setComment(
            "A candidate is matched to a truth particle whose mixed RecoToSim score is below this; one entry per "
            "decision point");
    desc.add<std::vector<double>>("maxEnergyResponseDeviation", {0.5})
        ->setComment("The energy step: |E_reco / E_true - 1| below this; one entry per decision point");
    return desc;
  }

  void ScoreMixer::mixAll(CandidateScores const& candidate, std::vector<MixedPair>& out) const {
    out.clear();
    out.reserve(candidate.pairs.size());
    for (auto const& pair : candidate.pairs)
      out.push_back(
          {pair.truthId, mix(candidate, pair, Direction::RecoToSim), mix(candidate, pair, Direction::SimToReco)});
  }

  std::string decisionPointLabel(double maxSimToRecoScore,
                                 double maxRecoToSimScore,
                                 double maxEnergyResponseDeviation) {
    auto format = [](double value) {
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.2f", value);
      std::string out(buffer);
      std::replace(out.begin(), out.end(), '.', 'p');
      return out;
    };
    return "simToReco" + format(maxSimToRecoScore) + "_recoToSim" + format(maxRecoToSimScore) + "_energy" +
           format(maxEnergyResponseDeviation);
  }

  std::vector<DecisionThresholds> decisionThresholds(edm::ParameterSet const& mixerConfig) {
    auto const name = mixerConfig.getParameter<std::string>("name");
    auto const simToReco = mixerConfig.getParameter<std::vector<double>>("maxSimToRecoScore");
    auto const recoToSim = mixerConfig.getParameter<std::vector<double>>("maxRecoToSimScore");
    auto const energy = mixerConfig.getParameter<std::vector<double>>("maxEnergyResponseDeviation");
    if (simToReco.empty() || simToReco.size() != recoToSim.size() || simToReco.size() != energy.size()) {
      throw cms::Exception("Configuration")
          << "score mixer '" << name
          << "': maxSimToRecoScore, maxRecoToSimScore and maxEnergyResponseDeviation must have the same, non-zero "
             "length, one entry per decision point; got "
          << simToReco.size() << ", " << recoToSim.size() << " and " << energy.size();
    }
    std::vector<DecisionThresholds> points;
    std::set<std::string> labels;
    for (std::size_t i = 0; i < simToReco.size(); ++i) {
      for (double const score : {simToReco[i], recoToSim[i]}) {
        if (!(score >= 0. && score <= 1.))
          throw cms::Exception("Configuration") << "score mixer '" << name << "', point " << i << ": score threshold "
                                                << score << " is outside [0, 1], where every score lies";
      }
      if (!(energy[i] > 0.))
        throw cms::Exception("Configuration") << "score mixer '" << name << "', point " << i
                                              << ": maxEnergyResponseDeviation must be positive, got " << energy[i];
      auto label = decisionPointLabel(simToReco[i], recoToSim[i], energy[i]);
      if (!labels.insert(label).second)
        throw cms::Exception("Configuration")
            << "score mixer '" << name << "': decision point " << i << " repeats the folder label " << label
            << "; the labels keep two decimals, so the points must differ at that precision";
      points.push_back({static_cast<float>(simToReco[i]),
                        static_cast<float>(recoToSim[i]),
                        static_cast<float>(energy[i]),
                        std::move(label)});
    }
    return points;
  }

  RecoDecision PFMatchDecider::decideReco(std::vector<MixedPair> const& pairs) const {
    RecoDecision decision;
    for (auto const& pair : pairs) {
      if (!pair.recoToSim || *pair.recoToSim >= thresholds_.maxRecoToSimScore)
        continue;
      ++decision.nMatched;
      if (decision.bestTruth < 0 || *pair.recoToSim < decision.bestScore) {
        decision.bestTruth = pair.truthId;
        decision.bestScore = *pair.recoToSim;
      }
    }
    return decision;
  }

  void PFMatchDecider::accumulateTruth(std::vector<MixedPair> const& pairs,
                                       int64_t candidateIndex,
                                       std::vector<TruthDecision>& perTruth) const {
    for (auto const& pair : pairs) {
      if (pair.truthId >= perTruth.size())
        continue;
      auto& decision = perTruth[pair.truthId];
      decision.touched = true;
      if (!pair.simToReco || *pair.simToReco >= thresholds_.maxSimToRecoScore)
        continue;
      ++decision.nMatched;
      if (decision.bestCandidate < 0 || *pair.simToReco < decision.bestScore) {
        decision.bestCandidate = candidateIndex;
        decision.bestScore = *pair.simToReco;
      }
    }
  }

  namespace {

    // The track score when the candidate has a track, the calorimeter score otherwise.
    // The particle-flow reading of a candidate: a charged candidate IS its track, and its
    // calorimeter deposits only refine its energy.
    class TrackFirstMixer : public ScoreMixer {
    public:
      using ScoreMixer::ScoreMixer;
      std::optional<float> mix(CandidateScores const& candidate, PairScores const& pair, Direction d) const override {
        return candidate.hasTrack ? pair.score(Constituent::Track, d) : pair.score(Constituent::Calo, d);
      }
    };

    // The calorimeter score when the candidate has calorimeter deposits, the track score
    // otherwise: a calorimeter-driven reading, in which a track only stands in when there
    // is no deposit at all, a muon say.
    class CaloFirstMixer : public ScoreMixer {
    public:
      using ScoreMixer::ScoreMixer;
      std::optional<float> mix(CandidateScores const& candidate, PairScores const& pair, Direction d) const override {
        return candidate.hasCalo ? pair.score(Constituent::Calo, d) : pair.score(Constituent::Track, d);
      }
    };

    // One constituent kind alone, whatever else the candidate has, so each associator can
    // be read on its own.
    class TrackOnlyMixer : public ScoreMixer {
    public:
      using ScoreMixer::ScoreMixer;
      std::optional<float> mix(CandidateScores const&, PairScores const& pair, Direction d) const override {
        return pair.score(Constituent::Track, d);
      }
    };

    class CaloOnlyMixer : public ScoreMixer {
    public:
      using ScoreMixer::ScoreMixer;
      std::optional<float> mix(CandidateScores const&, PairScores const& pair, Direction d) const override {
        return pair.score(Constituent::Calo, d);
      }
    };

    // The weighted mean of the scores the candidate has. A candidate with one constituent
    // kind gets that kind's score unchanged, so a photon is never penalised for its
    // missing track.
    class WeightedMixer : public ScoreMixer {
    public:
      explicit WeightedMixer(edm::ParameterSet const& cfg)
          : ScoreMixer(cfg),
            trackWeight_(cfg.getParameter<double>("trackWeight")),
            caloWeight_(cfg.getParameter<double>("caloWeight")) {
        if (trackWeight_ < 0. || caloWeight_ < 0. || trackWeight_ + caloWeight_ <= 0.)
          throw cms::Exception("Configuration")
              << "score mixer '" << name() << "': the weights must be non-negative and not both zero";
      }
      std::optional<float> mix(CandidateScores const&, PairScores const& pair, Direction d) const override {
        double sum = 0., weights = 0.;
        if (auto const s = pair.score(Constituent::Track, d); s && trackWeight_ > 0.) {
          sum += trackWeight_ * *s;
          weights += trackWeight_;
        }
        if (auto const s = pair.score(Constituent::Calo, d); s && caloWeight_ > 0.) {
          sum += caloWeight_ * *s;
          weights += caloWeight_;
        }
        if (weights <= 0.)
          return std::nullopt;
        return static_cast<float>(sum / weights);
      }

    private:
      const double trackWeight_;
      const double caloWeight_;
    };

    // The better of the available scores: a pair matches if EITHER constituent says so.
    class BestMixer : public ScoreMixer {
    public:
      using ScoreMixer::ScoreMixer;
      std::optional<float> mix(CandidateScores const&, PairScores const& pair, Direction d) const override {
        auto const track = pair.score(Constituent::Track, d);
        auto const calo = pair.score(Constituent::Calo, d);
        if (track && calo)
          return std::min(*track, *calo);
        return track ? track : calo;
      }
    };

    // The worse of the available scores: a pair matches only if EVERY constituent the
    // candidate has agrees, the strictest reading.
    class WorstMixer : public ScoreMixer {
    public:
      using ScoreMixer::ScoreMixer;
      std::optional<float> mix(CandidateScores const&, PairScores const& pair, Direction d) const override {
        auto const track = pair.score(Constituent::Track, d);
        auto const calo = pair.score(Constituent::Calo, d);
        if (track && calo)
          return std::max(*track, *calo);
        return track ? track : calo;
      }
    };

  }  // namespace

}  // namespace pftruth

// Example: a user-defined mixing. Uncomment, rebuild, and select it with type = "myMixer"
// in a mixer PSet of pfCandidateTruthValidation_cff. When both scores exist it weights the
// track by trackWeight and the calorimeter by its complement, so a single parameter moves
// the mixing from calorimeter-only (0) to track-only (1); with one score it uses that one.
// It reuses trackWeight, which scoreMixerDescription() already declares. A NEW parameter
// must be added there too, with a default, or every mixer PSet fails validation. Anything
// a mixing needs beyond the scores, the candidate kinematics say, is first added to
// CandidateScores and filled in PFCandidateTruthValidator::buildScores.
//
// namespace pftruth {
//   namespace {
//     class MyMixer : public ScoreMixer {
//     public:
//       explicit MyMixer(edm::ParameterSet const& cfg)
//           : ScoreMixer(cfg), trackWeight_(cfg.getParameter<double>("trackWeight")) {}
//       std::optional<float> mix(CandidateScores const& candidate, PairScores const& pair, Direction d) const override {
//         auto const track = pair.score(Constituent::Track, d);
//         auto const calo = pair.score(Constituent::Calo, d);
//         if (track && calo)  // both available: weight the track by trackWeight
//           return static_cast<float>(trackWeight_ * *track + (1. - trackWeight_) * *calo);
//         return track ? track : calo;  // one available: use it as it is; none: no match
//       }
//
//     private:
//       const double trackWeight_;
//     };
//   }  // namespace
// }  // namespace pftruth
// PFTRUTH_REGISTER_SCORE_MIXER("myMixer", pftruth::MyMixer);

PFTRUTH_REGISTER_SCORE_MIXER("trackFirst", pftruth::TrackFirstMixer);
PFTRUTH_REGISTER_SCORE_MIXER("caloFirst", pftruth::CaloFirstMixer);
PFTRUTH_REGISTER_SCORE_MIXER("trackOnly", pftruth::TrackOnlyMixer);
PFTRUTH_REGISTER_SCORE_MIXER("caloOnly", pftruth::CaloOnlyMixer);
PFTRUTH_REGISTER_SCORE_MIXER("weighted", pftruth::WeightedMixer);
PFTRUTH_REGISTER_SCORE_MIXER("best", pftruth::BestMixer);
PFTRUTH_REGISTER_SCORE_MIXER("worst", pftruth::WorstMixer);
