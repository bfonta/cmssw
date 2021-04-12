import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalrechitGPUAnalyzer = DQMEDAnalyzer('EcalRecHitGPUAnalyzer',
                                      cpuEERecHitCollection = cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
	                              cpuEBRecHitCollection = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
                                      gpuEERecHitCollection = cms.InputTag('ecalCPURecHitProducer', 'EcalRecHitsEE'),
                                      gpuEBRecHitCollection = cms.InputTag('ecalCPURecHitProducer', 'EcalRecHitsEB'),
)
