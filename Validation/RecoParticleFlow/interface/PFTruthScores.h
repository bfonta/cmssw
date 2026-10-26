#ifndef Validation_RecoParticleFlow_PFTruthScores_h
#define Validation_RecoParticleFlow_PFTruthScores_h

// The truth-graph scores of particle-flow candidates, kept SEPARATE per constituent kind.
//
// No PF associator exists and none is built here: a PF candidate is matched to the truth
// graph through the associators of its constituents, the track associator for its track
// and the PFCluster or trackster associators for its calorimeter deposits. Each pair
// (candidate, truth particle) carries up to four scores, two directions times two
// constituent kinds, all in [0, 1] and lower is better. How they are turned into one
// decision is the job of a ScoreMixer (PFScoreMixer.h), so the scores themselves never
// presuppose a mixing rule.
//
// A score the candidate CANNOT have is absent (std::nullopt): a photon has no track, a
// muon may have no calorimeter deposit. A score the candidate COULD have but whose
// constituent does not touch this truth particle is 1, the worst value, so a mixer can
// tell "no information" from "no match".

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace pftruth {

  // The particle-flow types the validation is binned in. The truth side derives them from
  // the pdg id and the charge, the reco side from the candidate's pdg id.
  enum class PFType : uint8_t { ChargedHadron = 0, Electron, Muon, Photon, NeutralHadron };
  inline constexpr std::size_t kNPFTypes = 5;
  inline constexpr std::array<const char*, kNPFTypes> kPFTypeNames = {
      {"chargedHadron", "electron", "muon", "photon", "neutralHadron"}};

  // A truth particle's type: e, mu and photon by species, every other species a hadron,
  // charged or neutral by its charge.
  [[nodiscard]] inline PFType pfTypeFromTruth(int32_t pdgId, double charge) {
    switch (std::abs(pdgId)) {
      case 11:
        return PFType::Electron;
      case 13:
        return PFType::Muon;
      case 22:
        return PFType::Photon;
      default:
        return charge != 0. ? PFType::ChargedHadron : PFType::NeutralHadron;
    }
  }

  // A candidate's type, from the pdg id reco::PFCandidate::translateTypeToPdgId gives it
  // (211, 11, 13, 22, 130, and 1 and 2 for the HF hadron and e/gamma). The HF types fold
  // into the neutral hadron and the photon, which is what they are. X (pdg id 0) has none.
  [[nodiscard]] inline std::optional<PFType> pfTypeFromReco(int32_t pdgId) {
    switch (std::abs(pdgId)) {
      case 211:
        return PFType::ChargedHadron;
      case 11:
        return PFType::Electron;
      case 13:
        return PFType::Muon;
      case 22:
      case 2:
        return PFType::Photon;
      case 130:
      case 1:
        return PFType::NeutralHadron;
      default:
        return std::nullopt;
    }
  }

  // The two directions of the association. RecoToSim is reco-normalised (how much of the
  // candidate the truth particle fails to explain) and drives fake and merge rates;
  // SimToReco is truth-normalised (how much of the truth particle the candidate fails to
  // cover) and drives efficiency and duplicate rates.
  enum class Direction : uint8_t { RecoToSim, SimToReco };

  // The constituent kinds a score comes from.
  enum class Constituent : uint8_t { Track, Calo };

  // Every score of one (candidate, truth particle) pair.
  struct PairScores {
    uint32_t truthId = 0;
    std::optional<float> trackRecoToSim;
    std::optional<float> trackSimToReco;
    std::optional<float> caloRecoToSim;
    std::optional<float> caloSimToReco;
    // Calorimetric energy the candidate shares with the truth particle, and the truth
    // particle's own energy in the calorimeter groups the candidate has deposits in.
    float caloSharedEnergy = 0.f;
    float caloTruthEnergy = 0.f;

    [[nodiscard]] std::optional<float> score(Constituent constituent, Direction direction) const {
      if (constituent == Constituent::Track)
        return direction == Direction::RecoToSim ? trackRecoToSim : trackSimToReco;
      return direction == Direction::RecoToSim ? caloRecoToSim : caloSimToReco;
    }
  };

  // Every truth particle one candidate touches through any of its constituents.
  struct CandidateScores {
    // The candidate has a track the track associator scored, or calorimeter constituents
    // one of the calorimetric associators scored. A constituent outside every configured
    // associated collection is not counted, so hasCalo can be false for a candidate that
    // does carry energy.
    bool hasTrack = false;
    bool hasCalo = false;
    std::vector<PairScores> pairs;
  };

}  // namespace pftruth

#endif
