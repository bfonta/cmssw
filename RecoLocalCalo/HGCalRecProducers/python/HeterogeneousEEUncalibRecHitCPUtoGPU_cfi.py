import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

EEUncalibRecHitCPUtoGPUProd = cms.EDProducer('EEUncalibRecHitCPUtoGPU',
                                 HGCEEUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCEEUncalibRecHits'), )
