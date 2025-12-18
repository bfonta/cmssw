import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# cmsRun <full_path_to>/ecalGeometryAnalyzer_cfg.py input=step2.root maxEvents=10
options = VarParsing.VarParsing('analysis')
options.register(
    'input', '',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Input file (only one supported)"
)
options.register(
    'output', 'data.root',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Output file."
)
options.parseArguments()

from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cff import recHitMapProducer as _recHitMapProducer

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.ProcessModifiers.enableCPfromPU_cff import enableCPfromPU
process = cms.Process("EcalGeometryAnalyzer",Phase2C17I13M9,enableCPfromPU)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.SimPhase2L1GlobalTriggerEmulator_cff')
process.load('L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTMenu_cff')
# process.load('HLTrigger.Configuration.HLT_75e33_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# process.GlobalTag.globaltag = '150X_mcRun4_realistic_v1'

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '') 

process.TFileService = cms.Service(
    "TFileService", 
    fileName = cms.string(options.output),
    closeFileFast = cms.untracked.bool(True)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
# process.MessageLogger.cerr.threshold = 'INFO'
# process.MessageLogger.cerr.INFO.limit = -1
# process.MessageLogger.debugModules = ["*"]

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:' + options.input)
)

ecalRecoClusters = "hltParticleFlowClusterECALUnseeded"

process.hltPFScAssocByEnergyScoreProducer = cms.EDProducer("BarrelPCToSCAssociatorByEnergyScoreProducer",
    hardScatterOnly = cms.bool(True),
    hitMapTag = cms.InputTag("hltRecHitMapProducer:barrelRecHitMap"),
    hits = cms.VInputTag("hltParticleFlowRecHitECALUnseeded", "hltParticleFlowRecHitHBHE"), # hltParticleFlowClusterHO
)

process.hltPFClusterSimClusterAssociationProducerECAL = cms.EDProducer("PCToSCAssociatorEDProducer",
    associator = cms.InputTag("hltPFScAssocByEnergyScoreProducer"),
    label_lcl = cms.InputTag(ecalRecoClusters),
    label_scl = cms.InputTag("mix","MergedCaloTruth")
)

process.ecalGeometryAnalyzer = cms.EDAnalyzer(
    'EcalGeometryAnalyzer',
    caloParticles = cms.InputTag("mix", "MergedCaloTruth"),
    recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    simHits = cms.InputTag("g4SimHits", "EcalHitsEB"),
    recClusters = cms.InputTag(ecalRecoClusters),
    simClusters = cms.InputTag("mix", "MergedCaloTruth"),
    clusterAssociator = cms.InputTag("hltPFClusterSimClusterAssociationProducerECAL"),
    enFracCut = cms.untracked.double(0.01),
    ptCut = cms.untracked.double(0.1),
    scoreCut = cms.untracked.double(1.0),
    responseCut = cms.untracked.double(0.6),
)

process.p = cms.Path(
    process.hltPFScAssocByEnergyScoreProducer
    * process.hltPFClusterSimClusterAssociationProducerECAL
    * process.ecalGeometryAnalyzer
)
