import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltBarrelSoARecHitsEB = cms.EDProducer("BarrelSoARecHitsProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    detector = cms.string('EB'),
    maxLayerIndex = cms.uint32(0),
    recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded")
)

hltBarrelSoARecHitsEBSerialSync = makeSerialClone(hltBarrelSoARecHitsEB)
