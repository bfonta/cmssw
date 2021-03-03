import FWCore.ParameterSet.Config as cms
import os, glob

def getHeterogeneousRecHitsSource(pu):
    indir = '/home/bfontana/'
    filename_suff = 'hadd_out_PU' + str(pu)
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

enableGPU = True
from Configuration.ProcessModifiers.gpu_cff import gpu

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HGCAL_noise_fC, HGCAL_chargeCollectionEfficiencies

#arguments parsing
from FWCore.ParameterSet.VarParsing import VarParsing
F = VarParsing('analysis')
F.register('PU',
           -1,
           F.multiplicity.singleton,
           F.varType.int,
           "Pileup to consider.")
F.parseArguments()

process = cms.Process("TESTgpu", gpu) if enableGPU else cms.Process("TESTnongpu")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')

process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string("histo.root"),
                                   closeFileFast = cms.untracked.bool(True)
                               )

    
process.source = getHeterogeneousRecHitsSource(F.PU)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )) #add option for edmStreams
process.options.numberOfThreads = 2
process.options.numberOfStreams = 2



process.Filler = cms.ESProducer("HeterogeneousHGCalHEFCellPositionsFiller")
process.Dummy = cms.EDProducer("BrunoDummyProducer",
                               Tok = cms.InputTag('HGCalUncalibRecHit','HGCHEFUncalibRecHits'))
process.task = cms.Task( process.Filler, process.Dummy )
process.path = cms.Path( process.task )

outkeeps = ['keep *_*_*_*']
process.out = cms.OutputModule( "PoolOutputModule", 
                                fileName = cms.untracked.string( '/eos/user/b/bfontana/GPUs/CellPositionsCheck__' + str(F.PU) + '.root'),
                                outputCommands = cms.untracked.vstring(outkeeps[0])
)
process.outpath = cms.EndPath(process.out)
