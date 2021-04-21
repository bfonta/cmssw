import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

PU=200

#package loading
process = cms.Process("binaryWriter")
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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
wantSummaryFlag = True
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( wantSummaryFlag )) #add option for edmStreams

process.BinaryWriter = cms.EDAnalyzer('BinaryWriter',
                                      HGCEEUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCEEUncalibRecHits'),
                                      fileName = cms.string('data.out'),
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


#process.consumer = cms.EDAnalyzer("GenericConsumer",                     
#                                  eventProducts = cms.untracked.vstring('EEFull') )

#process.consumer = cms.EDAnalyzer('GenericConsumer',
#                                  eventProducts = cms.untracked.vstring('recHitsClone') )
#eventProducts = cms.untracked.vstring('HGCalUncalibRecHit') ) #uncalib only (to assess reading speed)

process.path = cms.Path( process.BinaryWriter )

dirName = '.'
process.out = cms.OutputModule( "PoolOutputModule", 
                                fileName = cms.untracked.string( os.path.join(dirName, 'out.root') ),
)
process.outpath = cms.EndPath( process.out )
