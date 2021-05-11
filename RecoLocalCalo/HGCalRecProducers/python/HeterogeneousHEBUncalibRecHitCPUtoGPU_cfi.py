import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

HEBUncalibRecHitCPUtoGPUProd = cms.EDProducer('HEBUncalibRecHitCPUtoGPU',
                                 HGCHEBUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCHEBUncalibRecHits'), )
