import FWCore.ParameterSet.Config as cms

hltBarrelSoARecHitsHBProducer = cms.EDProducer("BarrelSoARecHitsProducer@alpaka",
    detector = cms.string("HB"),
    recHits = cms.InputTag("hltParticleFlowRecHitHBHE")
)
