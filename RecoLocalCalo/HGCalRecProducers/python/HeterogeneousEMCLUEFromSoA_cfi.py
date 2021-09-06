import FWCore.ParameterSet.Config as cms

EMCLUEFromSoAProd = cms.EDProducer('HGCalLayerClusterProducerEMFromSoA',
                                   EMCLUEHitsSoATok = cms.InputTag('EMCLUEGPUtoSoAProd', 'Hits'),
                                   EMCLUEClustersSoATok = cms.InputTag('EMCLUEGPUtoSoAProd', 'Clusters'))
