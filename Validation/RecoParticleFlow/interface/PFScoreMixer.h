#ifndef Validation_RecoParticleFlow_PFScoreMixer_h
#define Validation_RecoParticleFlow_PFScoreMixer_h

// Turning the separate track and calorimeter scores of a (candidate, truth particle) pair
// into one, and that one into a decision.
//
// The mixing is a strategy object, chosen by name from configuration, so several mixings
// can be compared side by side in one job and a new one is a class plus one registration
// line (see PFScoreMixer.cc), without touching the analyzer:
//
//   class MyMixer : public pftruth::ScoreMixer { ... };
//   PFTRUTH_REGISTER_SCORE_MIXER("myMixer", MyMixer);
//
// The DECISION is separate from the mixing. A mixer turns every pair of a candidate into
// a MixedPair once (ScoreMixer::mixAll), and a PFMatchDecider per threshold point applies
// its thresholds to those mixed pairs, so a threshold scan never mixes twice. Efficiency
// and duplicate rate always read the SimToReco direction, fake and merge rate the RecoToSim
// one.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Validation/RecoParticleFlow/interface/PFTruthScores.h"

namespace pftruth {

  // One pair's mixed scores, the only input of a decision.
  struct MixedPair {
    uint32_t truthId = 0;
    std::optional<float> recoToSim;
    std::optional<float> simToReco;
  };

  class ScoreMixer {
  public:
    explicit ScoreMixer(edm::ParameterSet const& cfg) : name_(cfg.getParameter<std::string>("name")) {}
    virtual ~ScoreMixer() = default;

    // The mixed score of one pair in one direction, in [0, 1] and lower is better, or
    // nullopt when the pair carries no score this mixing can use, which the decider reads
    // as "no match". The candidate is passed so a mixing can depend on what it has.
    [[nodiscard]] virtual std::optional<float> mix(CandidateScores const& candidate,
                                                   PairScores const& pair,
                                                   Direction direction) const = 0;

    // The label of the mixing in the DQM folder names.
    [[nodiscard]] std::string const& name() const { return name_; }

    // Both directions of every pair of a candidate, in pair order. out is overwritten.
    void mixAll(CandidateScores const& candidate, std::vector<MixedPair>& out) const;

  private:
    std::string name_;
  };

  using ScoreMixerFactory = std::function<std::unique_ptr<ScoreMixer>(edm::ParameterSet const&)>;

  // The registry of mixings by type name.
  std::map<std::string, ScoreMixerFactory>& scoreMixerRegistry();

  struct ScoreMixerRegistration {
    ScoreMixerRegistration(std::string const& type, ScoreMixerFactory factory) {
      scoreMixerRegistry()[type] = std::move(factory);
    }
  };

  // One mixing from its configuration PSet: "type" picks the class, "name" labels it, and
  // the remaining parameters belong to the class. An unknown type is a configuration
  // error that lists the known ones.
  [[nodiscard]] std::unique_ptr<ScoreMixer> makeScoreMixer(edm::ParameterSet const& cfg);

  // The description every mixer PSet is validated against: the union of the parameters
  // of the shipped mixings, each with its default.
  [[nodiscard]] edm::ParameterSetDescription scoreMixerDescription();

  // The thresholds of one decision point. One per score direction: the efficiency and
  // duplicate rate count a pair below maxSimToRecoScore as a match of the truth particle,
  // the fake and merge rate a pair below maxRecoToSimScore as a match of the candidate.
  // The energy step of both accepts |E_reco / E_true - 1| below maxEnergyResponseDeviation.
  struct DecisionThresholds {
    float maxSimToRecoScore = 0.5f;
    float maxRecoToSimScore = 0.5f;
    float maxEnergyResponseDeviation = 0.5f;
    // The folder label of the point, see decisionPointLabel.
    std::string label;
  };

  // The folder label of a decision point, e.g. "simToReco0p50_recoToSim0p50_energy0p50",
  // formatted from the CONFIGURED doubles with two decimals. The python configuration
  // builds the same string (pfCandidateTruthValidation_cff.pointLabel) from the same
  // doubles, so the harvesting finds the folders; formatting the float copies instead would
  // round some values differently (0.155 is 0.15 as a double and 0.16 as a float).
  [[nodiscard]] std::string decisionPointLabel(double maxSimToRecoScore,
                                               double maxRecoToSimScore,
                                               double maxEnergyResponseDeviation);

  // The decision points of one mixer PSet. Each threshold parameter is a vector and
  // point i takes entry i of every one of them, so a scan of the validation against the
  // thresholds is one mixer with longer vectors. The vectors must all have the same,
  // non-zero length, the scores must lie in [0, 1], the energy deviation must be positive
  // and two points may not share a label; anything else is a configuration error.
  [[nodiscard]] std::vector<DecisionThresholds> decisionThresholds(edm::ParameterSet const& mixerConfig);

  // What the decision says about one candidate, reco-driven.
  struct RecoDecision {
    // The best-scoring truth particle below threshold, -1 when none: a fake.
    int64_t bestTruth = -1;
    float bestScore = 1.f;
    // The number of truth particles below threshold; two or more is a merge.
    unsigned nMatched = 0;
  };

  // What the decision says about one truth particle, truth-driven, accumulated over the
  // candidates.
  struct TruthDecision {
    // The best-scoring candidate below threshold, -1 when none: an inefficiency.
    int64_t bestCandidate = -1;
    float bestScore = 1.f;
    // The number of candidates below threshold; two or more is a duplicate.
    unsigned nMatched = 0;
    // Whether any candidate carried any score for this particle at all, so a lost
    // particle can be told from one no constituent ever touched.
    bool touched = false;
  };

  // The decision at one threshold point, from the mixed pairs of one candidate. It knows
  // nothing about the mixing that produced them.
  class PFMatchDecider {
  public:
    explicit PFMatchDecider(DecisionThresholds thresholds) : thresholds_(std::move(thresholds)) {}

    [[nodiscard]] DecisionThresholds const& thresholds() const { return thresholds_; }

    // The reco-driven decision for one candidate.
    [[nodiscard]] RecoDecision decideReco(std::vector<MixedPair> const& pairs) const;

    // Adds one candidate's contribution to the truth-driven decisions, indexed by truth id.
    void accumulateTruth(std::vector<MixedPair> const& pairs,
                         int64_t candidateIndex,
                         std::vector<TruthDecision>& perTruth) const;

  private:
    DecisionThresholds thresholds_;
  };

}  // namespace pftruth

#define PFTRUTH_REGISTER_SCORE_MIXER_CONCAT2(a, b) a##b
#define PFTRUTH_REGISTER_SCORE_MIXER_CONCAT(a, b) PFTRUTH_REGISTER_SCORE_MIXER_CONCAT2(a, b)
#define PFTRUTH_REGISTER_SCORE_MIXER(TYPE_NAME, CLASS)                                                       \
  static const pftruth::ScoreMixerRegistration PFTRUTH_REGISTER_SCORE_MIXER_CONCAT(kRegistration, __LINE__)( \
      TYPE_NAME, [](edm::ParameterSet const& cfg) { return std::make_unique<CLASS>(cfg); })

#endif
