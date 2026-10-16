import FWCore.ParameterSet.Config as cms

hltBarrelLayerClustersEBL1Seeded = cms.EDProducer('BarrelLayerClusterProducer',
  recHits = cms.InputTag("hltParticleFlowRecHitECALL1Seeded"),
  plugin = cms.PSet(
    outlierDeltaFactor = cms.double(2.7 * 0.0175),
    kappa = cms.double(1),
    maxLayerIndex = cms.int32(0),
    deltac = cms.double(1.8 * 0.0175),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('EBCLUE')
  )
)

# With the alpaka modifier, the layer clusters converted from the SoA of the alpaka chain
# (hltBarrelSoARecHitsEBL1Seeded, hltBarrelCLUEsteringEBL1Seeded, hltBarrelSoALayerClustersEBL1Seeded) replace the CPU ones
_hltBarrelLayerClustersEBL1SeededFromSoA = cms.EDProducer("BarrelLayerClustersFromSoAProducer",
    detector = cms.string('EB'),
    barrelRecHitsLayerClustersSoA = cms.InputTag("hltBarrelCLUEsteringEBL1Seeded"),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsEBL1Seeded"),
    src = cms.InputTag("hltBarrelSoALayerClustersEBL1Seeded"),
    timeClname = cms.string('timeLayerCluster')
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
(ticl_barrel & alpaka).toReplaceWith(hltBarrelLayerClustersEBL1Seeded, _hltBarrelLayerClustersEBL1SeededFromSoA)
