import FWCore.ParameterSet.Config as cms

hltBarrelSoARecHitsEBL1Seeded = cms.EDProducer("BarrelSoARecHitsProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    detector = cms.string('EB'),
    maxLayerIndex = cms.uint32(0),
    recHits = cms.InputTag("hltParticleFlowRecHitECALL1Seeded")
)
