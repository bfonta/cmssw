import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# cmsRun <full_path_to>/ecalGeometryAnalyzer_cfg.py input=step2.root maxEvents=10
options = VarParsing.VarParsing('analysis')
options.register(
    'input', '',
    VarParsing.VarParsing.multiplicity.list,
    VarParsing.VarParsing.varType.string,
    "Input file(s)"
)
options.parseArguments()

process = cms.Process("EcalGeometryAnalyzer")

process.load('Configuration.Geometry.GeometryRecoDB_cff')
# process.load('Configuration.StandardSequences.GeometryRecoDB_cff')

process.TFileService = cms.Service(
    "TFileService", 
    fileName = cms.string("data.root"),
    closeFileFast = cms.untracked.bool(True)
)

process.CaloGeometryBuilder = cms.ESProducer(
    "CaloGeometryBuilder",                                                                         
    SelectedCalos = cms.vstring(
        'HCAL',
        'ZDC',
        # 'CASTOR', # missing the geometry
        'EcalBarrel',
        'EcalEndcap',
        'EcalPreshower',
        'TOWER'
    )
)

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = '150X_mcRun4_realistic_v1'

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

assert len(options.input) == 1
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:' + options.input[0])
)

process.ecalGeometryAnalyzer = cms.EDAnalyzer(
    'EcalGeometryAnalyzer',
    recHits = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    simHits = cms.InputTag("g4SimHits", "EcalHitsEB"),
    recClusters = cms.InputTag("hltParticleFlowClusterECAL"),
    simClusters = cms.InputTag("mix", "MergedCaloTruth")
)

process.p = cms.Path(process.ecalGeometryAnalyzer)
