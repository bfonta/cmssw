import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

EMCLUEGPUtoSoAProd = cms.EDProducer('HGCalLayerClusterProducerEMGPUtoSoA',
                                    EMInputCLUEGPU = cms.InputTag('EMCLUEGPUProd'),
)
