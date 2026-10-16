import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltBarrelSoALayerClustersHB = cms.EDProducer("BarrelSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    barrelRecHitsLayerClustersSoA = cms.InputTag("hltBarrelCLUEsteringHB"),
    barrelRecHitsSoA = cms.InputTag("hltBarrelSoARecHitsHB"),
    detector = cms.string('HB'),
    nHitsTime = cms.uint32(3)
)

hltBarrelSoALayerClustersHBSerialSync = makeSerialClone(hltBarrelSoALayerClustersHB,
    barrelRecHitsLayerClustersSoA = "hltBarrelCLUEsteringHBSerialSync",
    barrelRecHitsSoA = "hltBarrelSoARecHitsHBSerialSync"
)
