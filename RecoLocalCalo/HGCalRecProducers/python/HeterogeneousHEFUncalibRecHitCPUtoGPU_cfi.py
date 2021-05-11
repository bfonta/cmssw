import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

HEFUncalibRecHitCPUtoGPUProd = cms.EDProducer('HEFUncalibRecHitCPUtoGPU',
                                              HGCHEFUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCHEFUncalibRecHits'), )
