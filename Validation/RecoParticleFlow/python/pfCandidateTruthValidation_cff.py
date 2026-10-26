"""Particle-flow candidate validation against the MC-truth graph.

The reference is the pfReconstructable level of the graph. A candidate is matched through
the truth-graph associators of its constituents (SimGeneral/TruthGraphAssociatorProducers):
the track associator if it has a track, the PFCluster associators (barrel) or the trackster
associator (pfTICL) if it has calorimeter deposits. The track and calorimeter scores stay
separate; each entry of `mixers` turns them into one score with its own mixing type and its
own thresholds, which are equal-length vectors so a threshold scan is one entry. Every
(mixer, threshold point) gets its own folder; see the commented example below `mixers`.

Usage, on the reconstruction and on the harvesting step:

    cmsDriver.py ... --customise Validation/RecoParticleFlow/pfCandidateTruthValidation_cff.customisePFCandidateTruthValidation
"""

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from Validation.RecoParticleFlow.pfCandidateTruthValidator_cfi import pfCandidateTruthValidator as _validator

_dirName = "ParticleFlow/PFCandidateTruth"

# The particle-flow types, in the order of pftruth::kPFTypeNames.
pfTypes = ["chargedHadron", "electron", "muon", "photon", "neutralHadron"]

# The candidate collections. Phase-2 particle flow is the merge of the two: the barrel
# candidates reach their PFClusters through their PF block elements, the endcap ones are
# built one to one and in order from ticlCandidate, whose tracksters are their calorimeter
# deposits. The final particleFlow collection is not used: PFLinker drops e/gamma
# candidates without a GsfElectron or Photon, so its indices no longer line up.
candidateSources = cms.VPSet(
    cms.PSet(
        name=cms.string("barrel"),
        candidates=cms.InputTag("particleFlowTmpBarrel"),
        caloLink=cms.string("blocks"),
        ticlCandidates=cms.InputTag("ticlCandidate"),
    ),
    cms.PSet(
        name=cms.string("pfTICL"),
        candidates=cms.InputTag("pfTICL"),
        caloLink=cms.string("ticlCandidates"),
        ticlCandidates=cms.InputTag("ticlCandidate"),
    ),
)

# The calorimetric collections and the associators that scored them. Their labels must
# be among the ones truthGraphAssociationLabels_cff configures.
caloAssociations = cms.VPSet(
    cms.PSet(
        collection=cms.InputTag("particleFlowClusterECAL"),
        type=cms.string("PFCluster"),
        associator=cms.string("truthBranchPFClusterEcalAssociators"),
    ),
    cms.PSet(
        collection=cms.InputTag("particleFlowClusterHCAL"),
        type=cms.string("PFCluster"),
        associator=cms.string("truthBranchPFClusterHcalAssociators"),
    ),
    cms.PSet(
        collection=cms.InputTag("ticlCandidate"),
        type=cms.string("Trackster"),
        associator=cms.string("truthBranchTracksterAssociators"),
    ),
)


_thresholdNames = ("maxSimToRecoScore", "maxRecoToSimScore", "maxEnergyResponseDeviation")


def scoreMixer(name, type_, maxSimToRecoScore=(0.5,), maxRecoToSimScore=(0.5,), maxEnergyResponseDeviation=(0.5,),
               trackWeight=0.5, caloWeight=0.5):
    """One mixing of the track and calorimeter scores, with its decision thresholds.

    type_ picks the mixing (trackFirst, caloFirst, trackOnly, caloOnly, weighted, best,
    worst, or one registered by the user, see src/PFScoreMixer.cc); trackWeight and
    caloWeight are read by weighted only. Each threshold is a list, and point i takes entry
    i of all three, so a scan is one call with longer lists: they must have the same
    length, which the analyzer also enforces. Every point gets its own folder,
    Metrics/<name>/<pointLabel(...)>.
    """
    thresholds = dict(maxSimToRecoScore=list(maxSimToRecoScore), maxRecoToSimScore=list(maxRecoToSimScore),
                      maxEnergyResponseDeviation=list(maxEnergyResponseDeviation))
    lengths = {k: len(v) for k, v in thresholds.items()}
    if len(set(lengths.values())) != 1 or 0 in lengths.values():
        raise ValueError("score mixer '%s': the threshold lists must have the same, non-zero length, got %s"
                         % (name, lengths))
    return cms.PSet(
        name=cms.string(name),
        type=cms.string(type_),
        trackWeight=cms.double(trackWeight),
        caloWeight=cms.double(caloWeight),
        **{k: cms.vdouble(*v) for k, v in thresholds.items()},
    )


# The ways of turning the track and calorimeter scores into one, side by side, each at the
# default thresholds. A new mixing is a ScoreMixer subclass registered in
# Validation/RecoParticleFlow/src/PFScoreMixer.cc plus an entry here.
mixers = cms.VPSet(
    scoreMixer("TrackFirst", "trackFirst"),
    scoreMixer("CaloFirst", "caloFirst"),
    scoreMixer("Weighted", "weighted", trackWeight=0.5, caloWeight=0.5),
    scoreMixer("Best", "best"),
    scoreMixer("Worst", "worst"),
    scoreMixer("TrackOnly", "trackOnly"),
    scoreMixer("CaloOnly", "caloOnly"),
)

# Example: changing the mixing from a job configuration, after the customise has loaded
# this cff (for instance with --customise_commands). Each mixer is independent, so the
# examples combine freely.
#
#   from Validation.RecoParticleFlow.pfCandidateTruthValidation_cff import scoreMixer
#   validator = process.pfCandidateTruthValidator
#
#   # 1. Keep a single mixing and scan its thresholds: three decision points, entry i of
#   #    each list going together, so the three lists must have the same length.
#   validator.mixers = cms.VPSet(
#       scoreMixer("TrackFirst", "trackFirst",
#                  maxSimToRecoScore=[0.2, 0.5, 0.8],
#                  maxRecoToSimScore=[0.2, 0.5, 0.8],
#                  maxEnergyResponseDeviation=[0.5, 0.5, 0.5]))
#
#   # 2. Add a weighted mixing that trusts the track three times more than the calorimeter.
#   validator.mixers.append(scoreMixer("TrackHeavy", "weighted", trackWeight=0.75, caloWeight=0.25))
#
#   # 3. Change the mixing type of an existing entry, keeping its name and thresholds.
#   validator.mixers[0].type = "best"
#
#   # 4. Select a mixing written in C++ and registered as "myMixer" (see the commented
#   #    example in src/PFScoreMixer.cc).
#   validator.mixers.append(scoreMixer("Mine", "myMixer", trackWeight=0.7))
#
#   # The harvesting reads the folder list from the same configuration, so rebuild it on
#   # the harvesting step with the same mixers:
#   process.pfCandidateTruthHarvester.subDirs = metricFolders(validator)


def pointLabel(maxSimToRecoScore, maxRecoToSimScore, maxEnergyResponseDeviation):
    """The folder label of one decision point, the string DecisionThresholds::label builds."""
    fmt = lambda x: ("%.2f" % x).replace(".", "p")
    return "simToReco%s_recoToSim%s_energy%s" % (fmt(maxSimToRecoScore), fmt(maxRecoToSimScore),
                                                fmt(maxEnergyResponseDeviation))


def mixerPoints(mixer):
    """The folder labels of every decision point of one mixer PSet."""
    lists = [list(getattr(mixer, name)) for name in _thresholdNames]
    return [pointLabel(*point) for point in zip(*lists)]


pfCandidateTruthValidator = _validator.clone(
    candidateSources=candidateSources,
    caloAssociations=caloAssociations,
    mixers=mixers,
    dirName=_dirName,
)


def _sourceNames(sources):
    names = [s.name.value() for s in sources]
    return names + (["combined"] if len(names) > 1 else [])


def metricFolders(validator=pfCandidateTruthValidator):
    """The metric folders the validator books, in the layout of bookSource."""
    return [
        "%s/%s/Metrics/%s/%s/%s" % (validator.dirName.value(), source, mixer.name.value(), point, pfType)
        for source in _sourceNames(validator.candidateSources)
        for mixer in validator.mixers
        for point in mixerPoints(mixer)
        for pfType in pfTypes
    ]


_axes = (("pt", "p_{T} [GeV]"), ("eta", "#eta"), ("phi", "#phi"), ("energy", "E [GeV]"))


def _ratios(name, title, num, den):
    return ["%s_%s '%s vs %s;%s;%s' %s_%s %s_%s" % (name, axis, title, label, label, title, num, axis, den, axis)
            for axis, label in _axes]


_efficiency = (
    _ratios("eff", "efficiency", "num_assoc", "den")
    + _ratios("eff_pid", "efficiency with type", "num_pid", "den")
    + _ratios("eff_energy", "efficiency with type and energy", "num_energy", "den")
    + _ratios("duplicate_rate", "duplicate rate", "num_duplicate", "den")
    + _ratios("notTouched_rate", "untouched rate", "num_notTouched", "den")
    + _ratios("fake_rate", "fake rate", "num_fake_total", "den_fake")
    + _ratios("fake_rate_assoc", "fake rate, no match", "num_fake_assoc", "den_fake")
    + _ratios("fake_rate_pid", "fake rate, wrong type", "num_fake_pid", "den_fake")
    + _ratios("fake_rate_energy", "fake rate, wrong energy", "num_fake_energy", "den_fake")
    + _ratios("merge_rate", "merge rate", "num_merge", "den_fake")
)

pfCandidateTruthHarvester = DQMEDHarvester(
    "DQMGenericClient",
    subDirs=cms.untracked.vstring(*metricFolders()),
    efficiency=cms.vstring(*_efficiency),
    resolution=cms.vstring(),
    verbose=cms.untracked.uint32(0),
    outputFileName=cms.untracked.string(""),
)

pfCandidateTruthValidationSequence = cms.Sequence(pfCandidateTruthValidator)
pfCandidateTruthHarvestingSequence = cms.Sequence(pfCandidateTruthHarvester)


def customisePFCandidateTruthValidation(process):
    """Schedule the associators the PF validation reads and the validation itself on a
    reconstruction step, or its harvesting on a harvesting step. A no-op elsewhere."""
    from SimGeneral.TruthGraphAssociatorProducers import customiseTruthGraphAssociators as _truth

    if _truth._isHarvesting(process):
        process.load("Validation.RecoParticleFlow.pfCandidateTruthValidation_cff")
        _truth._appendPath(process, "pfCandidateTruthHarvestingPath", process.pfCandidateTruthHarvestingSequence)
        return process
    if not _truth._hasOfflineReco(process):
        return process

    _truth._loadAssociators(process)
    process.load("Validation.RecoParticleFlow.pfCandidateTruthValidation_cff")
    # Only the associators the validation reads; the vertex ones are not needed.
    _truth._appendPath(
        process,
        "pfCandidateTruthValidationPath",
        process.truthBranchTargets
        + process.allTrackToTruthBranchAssociators
        + process.truthBranchTracksterAssociators
        + process.truthBranchPFClusterEcalAssociators
        + process.truthBranchPFClusterHcalAssociators
        + process.pfCandidateTruthValidationSequence,
    )
    return process
