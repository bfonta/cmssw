import FWCore.ParameterSet.Config as cms
from functools import reduce

from HLTriggerOffline.JetMET.Validation.SingleJetValidation_cfi import *
from Validation.RecoJets.hltJetValidation_cff import *
from Validation.RecoMET.hltMETValidation_cff import *

def sumModules(alist):
    return reduce(lambda x, y: x + y, alist)

met_common_analyzers = [hltMetAnalyzerPF, hltMetAnalyzerPFCalo, hltMetTypeOneAnalyzerPFPuppi] 
met_run3_analyzers = met_common_analyzers + [hltMetAnalyzerPFNoMu]
met_ph2_analyzers =  met_common_analyzers + [hltMetAnalyzerPFPuppi]

##please do NOT include paths here!
HLTJetMETValSeq = cms.Sequence(
    SingleJetValidation
    + hltJetAnalyzerAK4PFPuppi
    + hltJetAnalyzerAK4PF
    + hltJetAnalyzerAK4PFCHS
    + sumModules(met_run3_analyzers)
)

_phase2_HLTJetMETValSeq = HLTJetMETValSeq.copyAndExclude(met_run3_analyzers)
_phase2_HLTJetMETValSeq += cms.Sequence(sumModules(met_ph2_analyzers))

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(HLTJetMETValSeq, _phase2_HLTJetMETValSeq)
