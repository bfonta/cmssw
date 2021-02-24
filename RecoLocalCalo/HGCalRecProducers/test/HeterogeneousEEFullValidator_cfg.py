import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

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
process.load('RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

#TFileService

dirName = '/eos/user/b/bfontana/Samples/'
fileName = 'validation' + str(F.PU) + '_EEFull.root'
process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string( os.path.join(dirName,fileName) ),
                                   closeFileFast = cms.untracked.bool(True)
                               )

process.source = getHeterogeneousRecHitsSource(F.PU)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )) #add option for edmStreams

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
                                weights        = HGCalRecHit.__dict__['layerWeights'] )

process.HGCalRecHits = HGCalRecHit.clone()

process.valid = cms.EDAnalyzer( 'HeterogeneousHGCalRecHitsValidator',
                                cpuRecHitsEEToken = cms.InputTag('HGCalRecHits', 'HGCEERecHits'),
                                gpuRecHitsEEToken = cms.InputTag('EEFull'),
                                cpuRecHitsHSiToken = cms.InputTag('HGCalRecHits', 'HGCHEFRecHits'),
                                gpuRecHitsHSiToken = cms.InputTag('EEFull'),
                                cpuRecHitsHSciToken = cms.InputTag('HGCalRecHits', 'HGCHEBRecHits'),
                                gpuRecHitsHSciToken = cms.InputTag('EEFull')
)

process.ee_t = cms.Task( process.EEFull )
process.gpu_t = cms.Task( process.ee_t )
process.cpu_t = cms.Task( process.HGCalRecHits )
process.path = cms.Path( process.valid, process.gpu_t, process.cpu_t )


process.out = cms.OutputModule( "PoolOutputModule", 
                                fileName = cms.untracked.string( os.path.join(dirName, 'out.root') ),
                                outputCommands = cms.untracked.vstring('drop *') )

process.outpath = cms.EndPath(process.out)
