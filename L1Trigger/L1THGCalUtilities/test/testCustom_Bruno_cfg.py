import FWCore.ParameterSet.Config as cms 

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
process = cms.Process('DIGI', Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
       #  fileNames = cms.untracked.vstring(''),
       #  fileNames = cms.untracked.vstring('/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/DoubleElectron_FlatPt-1To100/GEN-SIM-DIGI-RAW-MINIAOD/PU200_111X_mcRun4_realistic_T15_v1-v2/280000/003B8BCB-93B0-4040-854A-04C77E4BD066.root'),
                            #  fileNames = cms.untracked.vstring('/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/JPsiToMuMu_Pt0to100-pythia8_TuneCP5-gun/GEN-SIM-DIGI-RAW-MINIAOD/PU200_111X_mcRun4_realistic_T15_v1_ext1-v3/270000/01A1498B-3929-A04C-B651-8D418C11FABD.root'),
          fileNames = cms.untracked.vstring('/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/DoublePhoton_FlatPt-1To100/FEVT/NoPU_111X_mcRun4_realistic_T15_v1-v1/110000/001C5D05-8192-054B-827B-F1D34DFDDB96.root'),
       inputCommands=cms.untracked.vstring(
           'keep *',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
           'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016s_simEmtfDigis__HLT',
           )
       )

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SinglePhoton_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("testCustom_Bruno.root")
    )

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

# load HGCAL TPG simulation
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
#process.load('L1Trigger.L1THGCalUtilities.HGC3DClusterGenMatchSelector_cff')
process.load('L1Trigger.L1THGCalUtilities.hgcalTriggerNtuples_cff')

# Switch to latest trigger geometry containing information on links mapping /!\
# from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_decentralized_V11
# process = custom_geometry_decentralized_V11(process, links='signaldriven', implementation=2)
# If the above does not work
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V11_Imp3
process = custom_geometry_V11_Imp3(process)

process.hgcalTriggerGeometryESProducer.TriggerGeometry.JsonMappingFile = cms.FileInPath("L1Trigger/L1THGCal/data/hgcal_trigger_link_mapping_120links_v1.json")

# fill cluster layer info
process.ntuple_multiclusters.FillLayerInfo = True

from L1Trigger.L1THGCalUtilities.hgcalTriggerChains import HGCalTriggerChains
import L1Trigger.L1THGCalUtilities.vfe as vfe
import L1Trigger.L1THGCalUtilities.concentrator as concentrator
import L1Trigger.L1THGCalUtilities.clustering2d as clustering2d
import L1Trigger.L1THGCalUtilities.clustering3d as clustering3d
import L1Trigger.L1THGCalUtilities.selectors as selectors
import L1Trigger.L1THGCalUtilities.customNtuples as ntuple


chains = HGCalTriggerChains()
# Register algorithms
## VFE
chains.register_vfe("Floatingpoint", vfe.CreateVfe())
## ECON
chains.register_concentrator("Threshold", concentrator.CreateThreshold())
## BE1
chains.register_backend1("Dummy", clustering2d.CreateDummy())

# LP: relevant (new) part of producer - apply stage 1 truncation
#    | clustering2d.RozBinTruncation() handles the number of TCs per R/z bin
#    |    can take as input a distribution

ntcs_120links_default = [  2,  7, 27, 24, 19, 17, 16, 15, 14, 14, 13, 13,
                          13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10, 10,
                           9,  9, 10,  9,  9,  9,  8,  8,  7,  5,  3,  3,
                           3,  3,  3,  3,  3,  3]

ntcs_120links_flat =  [ 10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10, 
                        10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,
                        10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10, 
                        10,  10,  10,  10,  10,  10]

chains.register_backend1( "Truncation120default",  clustering2d.RozBinTruncation( maxTcsPerBin=ntcs_120links_default ))
chains.register_backend1( "Truncation120flat",     clustering2d.RozBinTruncation( maxTcsPerBin=ntcs_120links_flat    ))

## BE2
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import MAX_LAYERS
dr015 = [0.015]*(MAX_LAYERS+1)
chains.register_backend2("Histomaxxydr015", clustering3d.CreateHistoMaxXYVariableDr(distances=dr015))
chains.register_backend2("Histomax", clustering3d.CreateHistoMaxVariableDr())
chains.register_backend2("Histomaxnoareath20", clustering3d.CreateHistoMaxVariableDr(seed_threshold=20, seeds_norm_by_area=False))
# Register selector
# chains.register_selector("Genmatch", selectors.CreateGenMatch())


# Register ntuples
ntuple_list = ['event', 'gen', 'multiclusters', 'triggercells']
chains.register_ntuple("Genclustersntuple", ntuple.CreateNtuple(ntuple_list))

# Register trigger chains
stage1_algos = ['Dummy',
                'Truncation120default',
                'Truncation120flat']
stage2_algos = ['Histomax', 
                'Histomaxxydr015',
                'Histomaxnoareath20']

## Make cross product for BE algos
import itertools
for s1,s2 in itertools.product(stage1_algos,stage2_algos):
    chains.register_chain('Floatingpoint', 'Threshold', s1, s2, '', 'Genclustersntuple')	
    #chains.register_chain('Floatingpoint', 'Threshold', s1, s2, 'Genmatch', 'Genclustersntuple')

process = chains.create_sequences(process)

# Remove towers from sequence
process.hgcalTriggerPrimitives.remove(process.hgcalTowerMap)
process.hgcalTriggerPrimitives.remove(process.hgcalTower)

process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)
process.selector_step = cms.Path(process.hgcalTriggerSelector)
process.ntuple_step = cms.Path(process.hgcalTriggerNtuples)

# Schedule definition
process.schedule = cms.Schedule(process.hgcl1tpg_step, process.selector_step, process.ntuple_step)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
