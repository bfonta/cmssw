import FWCore.ParameterSet.Config as cms

from RecoTICL.LayerClustering.barrelLayerClusters_cfi import barrelLayerClusters as barrelLayerClusters_

barrelLayerClustersEB = barrelLayerClusters_.clone(
  recHits = 'particleFlowRecHitECAL',
  plugin = dict(
    outlierDeltaFactor = cms.double(2.7 * 0.0175),
    kappa = cms.double(1),
    maxLayerIndex = cms.int32(0),
    deltac = cms.double(1.8 * 0.0175),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('EBCLUE')
  )
)

barrelLayerClustersHB = barrelLayerClusters_.clone(
  recHits = 'particleFlowRecHitHBHE',
  plugin = dict(
    outlierDeltaFactor = cms.double(5 * 0.087),
    kappa = cms.double(0),
    maxLayerIndex = cms.int32(4),
    deltac = cms.double(3 * 0.087),
    fractionCutoff = cms.double(0.0),
    doSharing = cms.bool(False),
    type = cms.string('HBCLUE')
  )
)


#####################################################################
# CLUEstering layer-clustering chain (alpaka)
#
# The chain per detector is:
#   barrelSoARecHits<Det>       (PFRecHit  -> BarrelSoARecHits SoA)
#   barrelCLUEstering<Det>      (CLUE clustering on the SoA, in (eta, phi))
#   barrelSoALayerClusters<Det> (build the CaloCluster SoA)
#   barrelLayerClusters<Det>    (SoA -> legacy reco::CaloCluster products)
#
# Under the alpaka modifier, the last step replaces the CPU barrelLayerClusters<Det>.
# The CLUEstering parameters are the ones of the CPU plugins above. outlierDelta
# is an absolute distance, like the CPU outlierDeltaFactor.
#####################################################################
from RecoTICL.LayerClustering.barrelSoARecHitsProducer_cfi import barrelSoARecHitsProducer
from RecoTICL.LayerClustering.barrelCLUEsteringLayerClustersProducer_cfi import barrelCLUEsteringLayerClustersProducer
from RecoTICL.LayerClustering.barrelSoALayerClustersProducer_cfi import barrelSoALayerClustersProducer
from RecoTICL.LayerClustering.barrelLayerClustersFromSoAProducer_cfi import barrelLayerClustersFromSoAProducer

# ---- EB (ECAL barrel) ----
barrelSoARecHitsEB = barrelSoARecHitsProducer.clone(
    detector = 'EB',
    recHits = barrelLayerClustersEB.recHits.value(),
    maxLayerIndex = barrelLayerClustersEB.plugin.maxLayerIndex.value(),
    timeResolutionCalc = barrelLayerClustersEB.timeResolutionCalc.clone())
barrelCLUEsteringEB = barrelCLUEsteringLayerClustersProducer.clone(
    barrelRecHitsSoA = 'barrelSoARecHitsEB',
    deltac = barrelLayerClustersEB.plugin.deltac.value(),
    kappa = barrelLayerClustersEB.plugin.kappa.value(),
    outlierDelta = barrelLayerClustersEB.plugin.outlierDeltaFactor.value())
barrelSoALayerClustersEB = barrelSoALayerClustersProducer.clone(
    barrelRecHitsSoA = 'barrelSoARecHitsEB',
    barrelRecHitsLayerClustersSoA = 'barrelCLUEsteringEB',
    detector = 'EB',
    nHitsTime = barrelLayerClustersEB.nHitsTime.value())
_barrelLayerClustersFromSoAEB = barrelLayerClustersFromSoAProducer.clone(
    src = 'barrelSoALayerClustersEB',
    barrelRecHitsSoA = 'barrelSoARecHitsEB',
    barrelRecHitsLayerClustersSoA = 'barrelCLUEsteringEB',
    detector = 'EB',
    timeClname = barrelLayerClustersEB.timeClname.value())

# ---- HB (HCAL barrel) ----
barrelSoARecHitsHB = barrelSoARecHitsProducer.clone(
    detector = 'HB',
    recHits = barrelLayerClustersHB.recHits.value(),
    maxLayerIndex = barrelLayerClustersHB.plugin.maxLayerIndex.value(),
    timeResolutionCalc = barrelLayerClustersHB.timeResolutionCalc.clone())
barrelCLUEsteringHB = barrelCLUEsteringLayerClustersProducer.clone(
    barrelRecHitsSoA = 'barrelSoARecHitsHB',
    deltac = barrelLayerClustersHB.plugin.deltac.value(),
    kappa = barrelLayerClustersHB.plugin.kappa.value(),
    outlierDelta = barrelLayerClustersHB.plugin.outlierDeltaFactor.value())
barrelSoALayerClustersHB = barrelSoALayerClustersProducer.clone(
    barrelRecHitsSoA = 'barrelSoARecHitsHB',
    barrelRecHitsLayerClustersSoA = 'barrelCLUEsteringHB',
    detector = 'HB',
    nHitsTime = barrelLayerClustersHB.nHitsTime.value())
_barrelLayerClustersFromSoAHB = barrelLayerClustersFromSoAProducer.clone(
    src = 'barrelSoALayerClustersHB',
    barrelRecHitsSoA = 'barrelSoARecHitsHB',
    barrelRecHitsLayerClustersSoA = 'barrelCLUEsteringHB',
    detector = 'HB',
    timeClname = barrelLayerClustersHB.timeClname.value())

barrelLayerClustersEBAlpakaTask = cms.Task(barrelSoARecHitsEB, barrelCLUEsteringEB, barrelSoALayerClustersEB)
barrelLayerClustersHBAlpakaTask = cms.Task(barrelSoARecHitsHB, barrelCLUEsteringHB, barrelSoALayerClustersHB)

# With the alpaka modifier, the layer clusters from the SoA take over the labels of the CPU ones (consumed by
# hgcalMergeLayerClusters under ticl_barrel). The alpaka modules run on a GPU when one is available (e.g. selected
# with CUDA_VISIBLE_DEVICES), and on the CPU otherwise.
from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
(ticl_barrel & alpaka).toReplaceWith(barrelLayerClustersEB, _barrelLayerClustersFromSoAEB)
(ticl_barrel & alpaka).toReplaceWith(barrelLayerClustersHB, _barrelLayerClustersFromSoAHB)
