import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltBarrelSoALayerClustersEB = cms.EDProducer("BarrelSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    barrelRecHitsLayerClustersSoA = cms.InputTag("hltBarrelCLUEsteringEB"),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsEB"),
    detector = cms.string('EB'),
    nHitsTime = cms.uint32(3)
)

hltBarrelSoALayerClustersEBSerialSync = makeSerialClone(hltBarrelSoALayerClustersEB,
    barrelRecHitsLayerClustersSoA = "hltBarrelCLUEsteringEBSerialSync",
    barrelRecHitsSoA = "hltBarrelSoARecHitsEBSerialSync"
)
