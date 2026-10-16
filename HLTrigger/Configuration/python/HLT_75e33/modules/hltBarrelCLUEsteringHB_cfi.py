import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

# Same parameters as the CPU hltBarrelLayerClustersHB. outlierDelta is an absolute distance in (eta, phi), like the
# CPU outlierDeltaFactor, not a factor of deltac as in the HGCAL CLUEstering producer.
hltBarrelCLUEsteringHB = cms.EDProducer("BarrelCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsHB"),
    deltac = cms.float(3 * 0.087),
    kappa = cms.float(0),
    outlierDelta = cms.float(5 * 0.087)
)

hltBarrelCLUEsteringHBSerialSync = makeSerialClone(hltBarrelCLUEsteringHB,
    barrelRecHitsSoA = "hltBarrelSoARecHitsHBSerialSync"
)
