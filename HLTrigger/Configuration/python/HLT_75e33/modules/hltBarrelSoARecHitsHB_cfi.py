import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltBarrelSoARecHitsHB = cms.EDProducer("BarrelSoARecHitsProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    detector = cms.string('HB'),
    maxLayerIndex = cms.uint32(4),
    recHits = cms.InputTag("hltParticleFlowRecHitHBHE")
)

hltBarrelSoARecHitsHBSerialSync = makeSerialClone(hltBarrelSoARecHitsHB)
