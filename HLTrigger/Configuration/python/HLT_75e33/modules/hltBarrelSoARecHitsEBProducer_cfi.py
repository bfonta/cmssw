import FWCore.ParameterSet.Config as cms

hltBarrelSoARecHitsEBProducer = cms.EDProducer("BarrelSoARecHitsProducer@alpaka",
    detector = cms.string("EB"),
    recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded")
)
