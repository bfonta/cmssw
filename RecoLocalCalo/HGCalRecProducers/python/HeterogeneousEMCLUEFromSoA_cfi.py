import FWCore.ParameterSet.Config as cms

EMCLUEFromSoAProd = cms.EDProducer('EMCLUEFromSoA',
                                   EMCLUESoATok = cms.InputTag('EMCLUEGPUtoSoAProd'))
