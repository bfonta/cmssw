import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

PU=200
withGPU=1

#package loading
process = cms.Process("gpuTiming", gpu) if withGPU else cms.Process("cpuTiming")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')
process.load( "HLTrigger.Timer.FastTimerService_cfi" )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

indir =  '/home/bfontana/' #'/eos/user/b/bfontana/Samples/'
#filename_suff = 'hadd_out_PU' + str(PU) + '_uncompressed' #'step3_ttbar_PU' + str(PU)
filename_suff = 'hadd_out_PU' + str(PU) + '' #'step3_ttbar_PU' + str(PU)
fNames = [ 'file:' + x for x in glob.glob(os.path.join(indir, filename_suff + '*.root')) ]
if len(fNames)==0:
    print('Used globbing: ', glob.glob(os.path.join(indir, filename_suff + '*.root')))
    raise ValueError('No input files!')
print('Input: ', fNames)
keep = 'keep *'
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fNames),
                            inputCommands = cms.untracked.vstring(keep),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck") )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
wantSummaryFlag = True
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( wantSummaryFlag )) #add option for edmStreams

process.EEFull = cms.EDProducer('EERecHitFull',
                                HGCEEUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCEEUncalibRecHits'),
                                HGCEE_keV2DIGI = HGCalRecHit.__dict__['HGCEE_keV2DIGI'],
                                minValSiPar    = HGCalRecHit.__dict__['minValSiPar'],
                                maxValSiPar    = HGCalRecHit.__dict__['maxValSiPar'],
                                constSiPar     = HGCalRecHit.__dict__['constSiPar'],
                                noiseSiPar     = HGCalRecHit.__dict__['noiseSiPar'],
                                HGCEE_fCPerMIP = HGCalRecHit.__dict__['HGCEE_fCPerMIP'],
                                HGCEE_isSiFE   = HGCalRecHit.__dict__['HGCEE_isSiFE'],
                                HGCEE_noise_fC = HGCalRecHit.__dict__['HGCEE_noise_fC'],
                                HGCEE_cce      = HGCalRecHit.__dict__['HGCEE_cce'],
                                rcorr          = cms.vdouble( HGCalRecHit.__dict__['thicknessCorrection'][0:3] ),
                                weights        = HGCalRecHit.__dict__['layerWeights'],

                                fileName = cms.string('/eos/user/b/bfontana/GPUs/binaryPU' + str(PU) + '.out'),
                                nEvents = cms.uint32(100),
)

process.ThroughputService = cms.Service( "ThroughputService",
                                         eventRange = cms.untracked.uint32( 300 ),
                                         eventResolution = cms.untracked.uint32( 1 ),
                                         printEventSummary = cms.untracked.bool( wantSummaryFlag ),
                                         enableDQM = cms.untracked.bool( False )
                                         #valid only for enableDQM=True
                                         #dqmPath = cms.untracked.string( "HLT/Throughput" ),
                                         #timeRange = cms.untracked.double( 60000.0 ),
                                         #dqmPathByProcesses = cms.untracked.bool( False ),
                                         #timeResolution = cms.untracked.double( 5.828 )
)

process.FastTimerService.enableDQM = False
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.jsonFileName = 'resources.json'
###process.MessageLogger.categories.append('ThroughputService')

dirName = '/eos/user/b/bfontana/Samples/'
if withGPU:
    process.ee_task = cms.Task( process.EEFull )
    process.recHitsTask = cms.Task( process.ee_task )
    outkeeps = ['keep *_EEFull_*_*' ]
                #'keep *_HEFRecHitFromSoAProd_*_*',
                #'keep *_HEBRecHitFromSoAProd_*_*']
    process.out = cms.OutputModule( "PoolOutputModule", 
                                    fileName = cms.untracked.string( os.path.join(dirName, 'out.root') ),
                                    outputCommands = cms.untracked.vstring(outkeeps[0]) )

    #process.consumer = cms.EDAnalyzer("GenericConsumer",                     
    #                                  eventProducts = cms.untracked.vstring('EEFull') )

else:
    process.recHitsClone = HGCalRecHit.clone()
    process.recHitsTask = cms.Task( process.recHitsClone ) #CPU version
    outkeeps = ['keep *_*_' + f + '*_*' for f in ['HGCEERecHits', 'HGCHEFRecHits', 'HGCHEBRecHits'] ]
    process.out = cms.OutputModule( "PoolOutputModule", 
                                    fileName = cms.untracked.string( os.path.join(dirName, 'out.root') ),
                                    outputCommands = cms.untracked.vstring(outkeeps[0],
                                                                           outkeeps[1],
                                                                           outkeeps[2]) )

    #process.consumer = cms.EDAnalyzer('GenericConsumer',
    #                                  eventProducts = cms.untracked.vstring('recHitsClone') )
    #eventProducts = cms.untracked.vstring('HGCalUncalibRecHit') ) #uncalib only (to assess reading speed)


process.path = cms.Path( process.recHitsTask )
process.outpath = cms.EndPath(process.out)
