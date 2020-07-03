import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit

#package loading
process = cms.Process("gpuValidation", gpu) 
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2026D46Reco_cff')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi')
process.load('SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX_weights_v10
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#TFileService
fileName = 'validation.root'
process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True)
                               )


fNames = ['file:/afs/cern.ch/user/b/bfontana/CMSSW_11_1_0_pre6/src/20495.0_CloseByPGun_CE_E_Front_200um+CE_E_Front_200um_2026D41_GenSimHLBeamSpotFull+DigiFullTrigger_2026D41+RecoFullGlobal_2026D41+HARVESTFullGlobal_2026D41/step3.root']
keep = 'keep *'
drop = 'drop CSCDetIdCSCALCTPreTriggerDigiMuonDigiCollection_simCscTriggerPrimitiveDigis__HLT'
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fNames),
                            inputCommands = cms.untracked.vstring([keep, drop]),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"))

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )) #add option for edmStreams
process.HeterogeneousHGCalHEFRecHits = cms.EDProducer( 'HeterogeneousHGCalHEFRecHitProducer',
                                                       HGCHEFUncalibRecHitsTok = cms.InputTag('HGCalUncalibRecHit', 'HGCHEFUncalibRecHits'),
                                                       HGCHEF_keV2DIGI = HGCalRecHit.__dict__['HGCHEF_keV2DIGI'],
                                                       minValSiPar     = HGCalRecHit.__dict__['minValSiPar'],
                                                       maxValSiPar     = HGCalRecHit.__dict__['maxValSiPar'],
                                                       constSiPar      = HGCalRecHit.__dict__['constSiPar'],
                                                       noiseSiPar      = HGCalRecHit.__dict__['noiseSiPar'],
                                                       HGCHEF_fCPerMIP = HGCalRecHit.__dict__['HGCHEF_fCPerMIP'],
                                                       HGCHEF_isSiFE   = HGCalRecHit.__dict__['HGCHEF_isSiFE'],
                                                       HGCHEF_noise_fC = HGCalRecHit.__dict__['HGCHEF_noise_fC'],
                                                       HGCHEF_cce      = HGCalRecHit.__dict__['HGCHEF_cce'],
                                                       rangeMatch      = HGCalRecHit.__dict__['rangeMatch'],
                                                       rangeMask       = HGCalRecHit.__dict__['rangeMask'],
                                                       rcorr           = HGCalRecHit.__dict__['thicknessCorrection'],
                                                       weights         = HGCalRecHit.__dict__['layerWeights'] )


process.valid = cms.EDAnalyzer( "HeterogeneousHGCalRecHitsValidator",
                                cpuRecHitsHSiToken = cms.InputTag('HeterogeneousHGCalHEFRecHits','HeterogeneousHGCalHEFRecHits'),
                                gpuRecHitsHSiToken = cms.InputTag('HeterogeneousHGCalHEFRecHits','HeterogeneousHGCalHEFRecHits') )

process.recHitsTask = cms.Task( process.HeterogeneousHGCalHEFRecHits )
process.path = cms.Path( process.valid, process.recHitsTask )

process.out = cms.OutputModule( "PoolOutputModule", 
                                fileName = cms.untracked.string('out.root') )
process.outpath = cms.EndPath(process.out)
