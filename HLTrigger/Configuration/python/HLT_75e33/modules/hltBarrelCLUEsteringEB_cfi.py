import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

# Same parameters as the CPU hltBarrelLayerClustersEB. outlierDelta is an absolute distance in (eta, phi), like the
# CPU outlierDeltaFactor, not a factor of deltac as in the HGCAL CLUEstering producer.
hltBarrelCLUEsteringEB = cms.EDProducer("BarrelCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsEB"),
    deltac = cms.float(1.8 * 0.0175),
    kappa = cms.float(1),
    outlierDelta = cms.float(2.7 * 0.0175)
)

hltBarrelCLUEsteringEBSerialSync = makeSerialClone(hltBarrelCLUEsteringEB,
    barrelRecHitsSoA = "hltBarrelSoARecHitsEBSerialSync"
)
