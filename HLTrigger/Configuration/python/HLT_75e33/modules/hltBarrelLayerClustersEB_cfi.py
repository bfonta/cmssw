import FWCore.ParameterSet.Config as cms

hltBarrelLayerClustersEB = cms.EDProducer('BarrelLayerClusterProducer',
  recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
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
# (hltBarrelSoARecHitsEB, hltBarrelCLUEsteringEB, hltBarrelSoALayerClustersEB) replace the CPU ones
_hltBarrelLayerClustersEBFromSoA = cms.EDProducer("BarrelLayerClustersFromSoAProducer",
    detector = cms.string('EB'),
    barrelRecHitsLayerClustersSoA = cms.InputTag("hltBarrelCLUEsteringEB"),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsEB"),
    src = cms.InputTag("hltBarrelSoALayerClustersEB"),
    timeClname = cms.string('timeLayerCluster')
)

# Dedicated CPU ('SerialSync') instance, used by the alpakaValidationHLT GPU-vs-CPU comparison
hltBarrelLayerClustersEBSerialSync = _hltBarrelLayerClustersEBFromSoA.clone(
    barrelRecHitsLayerClustersSoA = "hltBarrelCLUEsteringEBSerialSync",
    barrelRecHitsSoA = "hltBarrelSoARecHitsEBSerialSync",
    src = "hltBarrelSoALayerClustersEBSerialSync"
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
(ticl_barrel & alpaka).toReplaceWith(hltBarrelLayerClustersEB, _hltBarrelLayerClustersEBFromSoA)
