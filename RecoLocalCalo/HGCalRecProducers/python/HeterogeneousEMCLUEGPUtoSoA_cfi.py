import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

EMCLUEGPUtoSoAProd = cms.EDProducer('HGCalLayerClusterProducerEMGPUtoSoA',
                                    EMInputCLUEHitsGPU = cms.InputTag('EMCLUEGPUProd', 'Hits'),
                                    EMInputCLUEClustersGPU = cms.InputTag('EMCLUEGPUProd', 'Clusters'),
)
