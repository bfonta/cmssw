import FWCore.ParameterSet.Config as cms

# Barrel (ECAL EB and HCAL HB) layer clustering.
#
# By default, the CPU BarrelLayerClusterProducer runs. With the ticl_barrel and alpaka
# process modifiers, each detector runs the CLUEstering chain instead:
#   hltBarrelSoARecHits<Det>       (PFRecHit -> BarrelSoARecHits SoA)
#   hltBarrelCLUEstering<Det>      (CLUEstering clustering, in (eta, phi))
#   hltBarrelSoALayerClusters<Det> (build the CaloCluster SoA)
#   hltBarrelLayerClusters<Det>    (SoA -> legacy reco::CaloCluster)
#
from ..modules.hltBarrelSoARecHitsEB_cfi import *
from ..modules.hltBarrelCLUEsteringEB_cfi import *
from ..modules.hltBarrelSoALayerClustersEB_cfi import *
from ..modules.hltBarrelLayerClustersEB_cfi import *
from ..modules.hltBarrelSoARecHitsHB_cfi import *
from ..modules.hltBarrelCLUEsteringHB_cfi import *
from ..modules.hltBarrelSoALayerClustersHB_cfi import *
from ..modules.hltBarrelLayerClustersHB_cfi import *
from ..modules.hltBarrelSoARecHitsEBL1Seeded_cfi import *
from ..modules.hltBarrelCLUEsteringEBL1Seeded_cfi import *
from ..modules.hltBarrelSoALayerClustersEBL1Seeded_cfi import *
from ..modules.hltBarrelLayerClustersEBL1Seeded_cfi import *

HLTBarrelLayerClustersSequence = cms.Sequence(
        hltBarrelLayerClustersEB+
        hltBarrelLayerClustersHB)

HLTBarrelLayerClustersL1SeededSequence = cms.Sequence(
        hltBarrelLayerClustersEBL1Seeded)

# Dedicated CPU ('SerialSync') instances, used by the alpakaValidationHLT
# GPU-vs-CPU comparison. Empty without the alpaka chain.
HLTBarrelLayerClustersSequenceSerialSync = cms.Sequence()

_HLTBarrelLayerClustersSequence_alpaka = cms.Sequence(
        hltBarrelSoARecHitsEB+
        hltBarrelCLUEsteringEB+
        hltBarrelSoALayerClustersEB+
        hltBarrelLayerClustersEB+
        hltBarrelSoARecHitsHB+
        hltBarrelCLUEsteringHB+
        hltBarrelSoALayerClustersHB+
        hltBarrelLayerClustersHB)

_HLTBarrelLayerClustersL1SeededSequence_alpaka = cms.Sequence(
        hltBarrelSoARecHitsEBL1Seeded+
        hltBarrelCLUEsteringEBL1Seeded+
        hltBarrelSoALayerClustersEBL1Seeded+
        hltBarrelLayerClustersEBL1Seeded)

_HLTBarrelLayerClustersSequenceSerialSync_alpaka = cms.Sequence(
        hltBarrelSoARecHitsEBSerialSync+
        hltBarrelCLUEsteringEBSerialSync+
        hltBarrelSoALayerClustersEBSerialSync+
        hltBarrelLayerClustersEBSerialSync+
        hltBarrelSoARecHitsHBSerialSync+
        hltBarrelCLUEsteringHBSerialSync+
        hltBarrelSoALayerClustersHBSerialSync+
        hltBarrelLayerClustersHBSerialSync)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
(ticl_barrel & alpaka).toReplaceWith(HLTBarrelLayerClustersSequence, _HLTBarrelLayerClustersSequence_alpaka)
(ticl_barrel & alpaka).toReplaceWith(HLTBarrelLayerClustersL1SeededSequence, _HLTBarrelLayerClustersL1SeededSequence_alpaka)
(ticl_barrel & alpaka).toReplaceWith(HLTBarrelLayerClustersSequenceSerialSync, _HLTBarrelLayerClustersSequenceSerialSync_alpaka)
