import os, sys, glob
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#arguments parsing
from FWCore.ParameterSet.VarParsing import VarParsing
F = VarParsing('analysis')
F.register('pu',
           0,
           F.multiplicity.singleton,
           F.varType.bool,
           "Whether to run with pile-up.")
F.register('fidx',
           0,
           F.multiplicity.singleton,
           F.varType.int,
           "Which file index to consider.")
F.register('mask',
           -1,
           F.multiplicity.singleton,
           F.varType.int,
           "Mask to be used. Accepted values: 3, 4, 5 or 6.")
F.register('samples',
           '',
           F.multiplicity.singleton,
           F.varType.string,
           'Which samples to use. Inner ("inner"), outer ("outer"), or both ("all").')
F.parseArguments()
print("********************")
print("Input arguments:")
for k,v in F.__dict__["_singletons"].items():
    print("{}: {}".format(k,v))
    print("********************")

#package loading
process = cms.Process("postRECO", eras.Phase2C8)
process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.Geometry.GeometryExtended2023D28_cff')
process.load('Configuration.Geometry.GeometryExtended2023D28Reco_cff')
"""
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring(
                                        'detailedInfo'),
                                    detailedInfo = cms.untracked.PSet(
                                        threshold = cms.untracked.string('INFO'),
                                        default = cms.untracked.PSet(
                                            limit = cms.untracked.int32(-1))),
                                    debugModules = cms.untracked.vstring('*'))
"""

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX_weights_v10

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

indir1 = "/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomEGunProducer_PionGun_SciStudies_bfontana_inner_20190716/RECO/"
indir2 = "/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomEGunProducer_PionGun_SciStudies_bfontana_outer_20190716/RECO/"
indir3 = "/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomEGunProducer_PionGun_CalibrationStudies_bfontana_central_20190822/RECO/"

glob1 = glob.glob(os.path.join(indir1,"*.root"))
glob2 = glob.glob(os.path.join(indir2,"*.root"))
glob3 = glob.glob(os.path.join(indir3,"*.root"))
if F.samples == 'all':
    glob_tot = glob1 + glob2
elif F.samples == 'inner':
    glob_tot = glob1
elif F.samples == 'outer':
    glob_tot = glob2
elif F.samples == 'central':
    glob_tot = glob3
else:
    raise ValueError('Insert a valid "samples" option!')
print("Total number of files: ", len(glob_tot))
fNames = ["file:" + it for it in glob_tot][F.fidx]
print("this file: ", fNames)

if isinstance(fNames,list):     
    process.source = cms.Source("PoolSource",
                        fileNames = cms.untracked.vstring(fNames),
                        duplicateCheckMode = cms.untracked.string("noDuplicateCheck"))
else:
    process.source = cms.Source("PoolSource",
                        fileNames = cms.untracked.vstring(fNames),
                        duplicateCheckMode = cms.untracked.string("noDuplicateCheck"))

process.RecHitsMasked = cms.EDProducer('HGCalMaskProd',
                            recHitsCEEToken = cms.InputTag('HGCalRecHit', 'HGCEERecHits'),
                            recHitsHSiToken = cms.InputTag('HGCalRecHit', 'HGCHEFRecHits'),
                            Mask = cms.uint32(F.mask))

process.an = cms.EDAnalyzer("HGCalMaskResolutionAna",
                            recHitsCEEToken = cms.InputTag('HGCalRecHit','HGCEERecHits'),
                            recHitsHSiToken = cms.InputTag('HGCalRecHit','HGCHEFRecHits'),
                            recHitsHScToken = cms.InputTag('HGCalRecHit','HGCHEBRecHits'),
                            thicknessCorrection = cms.vdouble(0.781, 0.776, 0.769),
                            dEdXWeights = dEdX_weights_v10,
                            geometrySource = cms.vstring('HGCalEESensitive',
                                                         'HGCalHESiliconSensitive',
                                                         'HGCalHEScintillatorSensitive'),
                            distancesSR1 = cms.vdouble(30., 30., 30.),
                            distancesSR2 = cms.vdouble(40., 40., 40.),
                            distancesSR3 = cms.vdouble(50., 50., 50.),
                            nControlRegions = cms.int32(2), #5 for photons
                            particle = cms.string('pion'),
                            byClosest = cms.bool(False))

process.an_mask = process.an.clone(
    recHitsCEEToken = cms.InputTag("RecHitsMasked", "HGCEERecHits"),
    recHitsHSiToken = cms.InputTag("RecHitsMasked", "HGCHSiRecHits"))

pu_str = "pu" if F.pu else "nopu"
fileName = str(F.fidx)+"_mask"+str(F.mask)+"_"+F.samples+"_"+pu_str
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName+".root"))
#fileName = fileName + '_out'
#process.out = cms.OutputModule("PoolOutputModule", 
                               #fileName = cms.untracked.string(fileName+".root"))
process.p = cms.Path(process.RecHitsMasked * process.an * process.an_mask)
#process.outpath = cms.EndPath(process.out)
