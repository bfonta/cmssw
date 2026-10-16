import FWCore.ParameterSet.Config as cms

hltBarrelSoALayerClustersEBL1Seeded = cms.EDProducer("BarrelSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    barrelRecHitsLayerClustersSoA = cms.InputTag("hltBarrelCLUEsteringEBL1Seeded"),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsEBL1Seeded"),
    detector = cms.string('EB'),
    nHitsTime = cms.uint32(3)
)
