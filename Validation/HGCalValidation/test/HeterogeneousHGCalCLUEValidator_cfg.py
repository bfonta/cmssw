import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters

def getHeterogeneousRecHitsSource(pu):
    indir = '/eos/user/b/bfontana/Samples/' #indir = '/home/bfontana/'
    filename_suff = 'step3_ttbar_PU' + str(pu) #filename_suff = 'hadd_out_PU' + str(pu)
    fNames = [ 'file:' + x for x in glob.glob(os.path.join(indir, filename_suff + '*.root')) ]
    print(indir, filename_suff, pu, fNames)
    for _ in range(4):
        fNames.extend(fNames)
    if len(fNames)==0:
        print('Used globbing: ', glob.glob(os.path.join(indir, filename_suff + '*.root')))
        raise ValueError('No input files!')

    keep = 'keep *'
    drop1 = 'drop CSCDetIdCSCALCTPreTriggerDigiMuonDigiCollection_simCscTriggerPrimitiveDigis__HLT'
    drop2 = 'drop HGCRecHitsSorted_HGCalRecHit_HGC*E*RecHits_*'
    return cms.Source("PoolSource",
                      fileNames = cms.untracked.vstring(fNames),
                      inputCommands = cms.untracked.vstring(keep, drop1, drop2),
                      duplicateCheckMode = cms.untracked.string("noDuplicateCheck"))

#arguments parsing
from FWCore.ParameterSet.VarParsing import VarParsing
F = VarParsing('analysis')
F.register('PU',
           1,
           F.multiplicity.singleton,
           F.varType.int,
           "Pileup to consider.")
F.parseArguments()

#package loading
process = cms.Process("gpuValidation", gpu) 
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')

# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

#TFileService

dirName = '/eos/user/b/bfontana/Samples/'
fileName = 'validationCLUE' + str(F.PU) + '.root'
process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string( os.path.join(dirName,fileName) ),
                                   closeFileFast = cms.untracked.bool(True)
                               )

process.source = getHeterogeneousRecHitsSource(F.PU)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )) #add option for edmStreams

# Filling positions conditions data
process.HeterogeneousHGCalPositionsFiller = cms.ESProducer("HeterogeneousHGCalPositionsFiller")

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitGPU_cfi')
#add this: process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEMCLUEGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEMCLUEGPUtoSoA_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEMCLUEFromSoA_cfi')

process.HGCalRecHit = HGCalRecHit.clone()
process.hgcalLayerClusters = hgcalLayerClusters.clone()

process.valid = cms.EDAnalyzer( 'HeterogeneousHGCalCLUEValidator',
                                cpuHitsEMToken = cms.InputTag('hgcalLayerClusters'),
                                gpuHitsEMToken = cms.InputTag('EMCLUEGPUtoSoAProd', 'Hits'),
                                cpuClustersEMToken = cms.InputTag('hgcalLayerClusters'),
                                gpuClustersEMToken = cms.InputTag('EMCLUEFromSoAProd', 'Clusters'))

process.em_task = cms.Task( process.EERecHitGPUProd,
                            process.EMCLUEGPUProd, process.EMCLUEGPUtoSoAProd, process.EMCLUEFromSoAProd,
)

process.gpu_t = cms.Task( process.HeterogeneousHGCalPositionsFiller,
                          process.em_task
)

process.cpu_t = cms.Task( process.HGCalRecHit, process.hgcalLayerClusters )

process.path = cms.Path( process.valid, process.gpu_t, process.cpu_t )

# process.consumer = cms.EDAnalyzer("GenericConsumer",                     
#                                   eventProducts = cms.untracked.vstring('EMCLUEFromSoAProd',) )
# process.consume_step = cms.EndPath(process.consumer)

process.out = cms.OutputModule( "PoolOutputModule", 
                                fileName = cms.untracked.string( os.path.join(dirName, 'out.root') ),
                                outputCommands = cms.untracked.vstring('drop *') )
process.outpath = cms.EndPath(process.out)
