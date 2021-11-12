import FWCore.ParameterSet.Config as cms
import os, glob
from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import HGCAL_noise_fC, HGCAL_chargeCollectionEfficiencies

def getHeterogeneousRecHitsSource(pu):
    indir = '/eos/user/b/bfontana/Samples/'
    #indir = '/home/scratch/'
    #filename_suff = 'step3_ttbar_PU' + str(pu)
    filename_suff = 'hadd_out_PU' + str(pu)
    
    fNames = [ 'file:' + x for x in glob.glob(os.path.join(indir, filename_suff + '*.root')) ]
    print("PU={}".format(pu))
    print(fNames)
    
    for _ in range(4):
        fNames.extend(fNames)
    if len(fNames)==0:
        print('Used globbing: ', glob.glob(os.path.join(indir, filename_suff + '*.root')))
        raise ValueError('No input files!')

    keep = 'keep *'
    drop1 = 'drop CSCDetIdCSCALCTPreTriggerDigiMuonDigiCollection_simCscTriggerPrimitiveDigis__HLT'
    drop2 = 'drop HGCRecHitsSorted_HGCalRecHit_HGC*E*RecHits_*'

    # process.source = cms.Source("RepeatingCachedRootSource",
    #                             fileNames = cms.untracked.vstring(fNames),
    #                             inputCommands = cms.untracked.vstring(keep, drop1, drop2),
    #                             repeatNEvents = cms.untracked.uint32(47))
    return cms.Source("PoolSource",
                      fileNames = cms.untracked.vstring(fNames),
                      inputCommands = cms.untracked.vstring(keep, drop1, drop2),
                      duplicateCheckMode = cms.untracked.string("noDuplicateCheck"))

PU=0
enableGPU = True

process = cms.Process("gpuTiming", gpu) if enableGPU else cms.Process("cpuTiming")
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string("histo.root"),
                                   closeFileFast = cms.untracked.bool(True)
                               )

process.source = getHeterogeneousRecHitsSource(PU)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
wantSummaryFlag = True
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( wantSummaryFlag )) #add option for edmStreams

# Filling positions conditions data
process.HeterogeneousHGCalPositionsFiller = cms.ESProducer("HeterogeneousHGCalPositionsFiller")

process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEERecHitGPU_cfi')
#process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousHEFRecHitGPU_cfi')
process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEMCLUEGPU_cfi')
#process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEMCLUEGPUtoSoA_cfi')
#process.load('RecoLocalCalo.HGCalRecProducers.HeterogeneousEMCLUEFromSoA_cfi')

process.load( "HLTrigger.Timer.FastTimerService_cfi" )
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

process.ee_task = cms.Task( process.EERecHitGPUProd, #process.HEFRecHitGPUProd,
                            #process.EMCLUEGPUProd, process.EMCLUEGPUtoSoAProd, process.EMCLUEFromSoAProd,
                            process.EMCLUEGPUProd,
)

process.global_task = cms.Task( process.HeterogeneousHGCalPositionsFiller,
                                process.ee_task
)
process.path = cms.Path( process.global_task )

process.consumer = cms.EDAnalyzer("GenericConsumer",
                                  eventProducts = cms.untracked.vstring('EMCLUEGPUProd',) )
process.consume_step = cms.EndPath(process.consumer)
