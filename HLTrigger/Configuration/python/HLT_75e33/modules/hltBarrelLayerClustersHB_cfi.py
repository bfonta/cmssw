import FWCore.ParameterSet.Config as cms

hltBarrelLayerClustersHB = cms.EDProducer('BarrelLayerClusterProducer',
  recHits = cms.InputTag("hltParticleFlowRecHitHBHE"),
  plugin = cms.PSet(
    outlierDeltaFactor = cms.double(5 * 0.087),
    kappa = cms.double(0),
    maxLayerIndex = cms.int32(4),
    deltac = cms.double(3 * 0.087),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('HBCLUE')
  )
)

# With the alpaka modifier, the layer clusters converted from the SoA of the alpaka chain
# (hltBarrelSoARecHitsHB, hltBarrelCLUEsteringHB, hltBarrelSoALayerClustersHB) replace the CPU ones
_hltBarrelLayerClustersHBFromSoA = cms.EDProducer("BarrelLayerClustersFromSoAProducer",
    detector = cms.string('HB'),
    barrelRecHitsLayerClustersSoA = cms.InputTag("hltBarrelCLUEsteringHB"),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsHB"),
    src = cms.InputTag("hltBarrelSoALayerClustersHB"),
    timeClname = cms.string('timeLayerCluster')
)

# Dedicated CPU ('SerialSync') instance, used by the alpakaValidationHLT GPU-vs-CPU comparison
hltBarrelLayerClustersHBSerialSync = _hltBarrelLayerClustersHBFromSoA.clone(
    barrelRecHitsLayerClustersSoA = "hltBarrelCLUEsteringHBSerialSync",
    barrelRecHitsSoA = "hltBarrelSoARecHitsHBSerialSync",
    src = "hltBarrelSoALayerClustersHBSerialSync"
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
(ticl_barrel & alpaka).toReplaceWith(hltBarrelLayerClustersHB, _hltBarrelLayerClustersHBFromSoA)
