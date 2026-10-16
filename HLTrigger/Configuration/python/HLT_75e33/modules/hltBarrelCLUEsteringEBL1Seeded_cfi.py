import FWCore.ParameterSet.Config as cms

# Same parameters as the CPU hltBarrelLayerClustersEBL1Seeded. outlierDelta is an absolute distance in (eta, phi), like the
# CPU outlierDeltaFactor, not a factor of deltac as in the HGCAL CLUEstering producer.
hltBarrelCLUEsteringEBL1Seeded = cms.EDProducer("BarrelCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsEBL1Seeded"),
    deltac = cms.float(1.8 * 0.0175),
    kappa = cms.float(1),
    outlierDelta = cms.float(2.7 * 0.0175)
)
