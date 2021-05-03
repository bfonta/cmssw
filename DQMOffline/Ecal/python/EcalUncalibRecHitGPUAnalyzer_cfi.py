import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalUncalibRechitGPUAnalyzer = DQMEDAnalyzer('EcalUncalibRecHitGPUAnalyzer',
                                             cpuEEUncalibRecHitCollection = cms.InputTag('ecalMultiFitUncalibRecHit',
                                                                                         'EcalUncalibRecHitsEE'),
	                                     cpuEBUncalibRecHitCollection = cms.InputTag('ecalMultiFitUncalibRecHit',
                                                                                         'EcalUncalibRecHitsEB'),
                                             gpuEEUncalibRecHitCollection = cms.InputTag('ecalCPUUncalibRecHitProducer',
                                                                                         'EcalUncalibRecHitsEE'),
                                             gpuEBUncalibRecHitCollection = cms.InputTag('ecalCPUUncalibRecHitProducer',
                                                                                         'EcalUncalibRecHitsEB'),
)
