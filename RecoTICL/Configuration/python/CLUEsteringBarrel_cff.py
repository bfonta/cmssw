import FWCore.ParameterSet.Config as cms

from RecoTICL.LayerClustering.hgCalSoAMergeLayerClustersProducer_cfi import hgCalSoAMergeLayerClustersProducer as _hgCalSoAMergeLayerClustersProducer
from RecoTICL.LayerClustering.hgcalLayerClusters_cff import hgcalMergeLayerClusters as _hgcalMergeLayerClusters
from RecoTICL.PatternRecognition.barrelCLUEsteringTrackstersProducer_cfi import barrelCLUEsteringTrackstersProducer as _barrelCLUEsteringTrackstersProducer
from RecoTICL.PatternRecognition.hgCalSoATrackstersProducer_cfi import hgCalSoATrackstersProducer as _hgCalSoATrackstersProducer
from RecoTICL.PatternRecognition.hgCalTrackstersFromSoAProducer_cfi import hgCalTrackstersFromSoAProducer as _hgCalTrackstersFromSoAProducer

# The layer clusters of EB and HB, merged on the device by barrelSoAMergeLayerClusters, are clustered into tracksters
# with CLUEstering, and the tracksters are built on the device

# The barrel calorimeters in the order in which their layer clusters are merged, on the device and by
# hgcalMergeLayerClusters (ticl_barrel), where they follow the layer clusters of HGCAL: the two merged collections then
# index the same barrel layer clusters, once shifted by the number of the HGCAL ones
_barrelComponents = ['EB', 'HB']
_barrelLayerClusters = ['barrelLayerClusters' + component for component in _barrelComponents]

# The barrel layer clusters are the extra ones of hgcalMergeLayerClusters (set by ticl_barrel), merged after those of
# the HGCAL SoA layer clusters: they must come first among the extra ones, in the order of _barrelComponents. The
# modifiers of the process apply when a cff is imported, so the import above already sees them. A later customisation
# is not checked here, but the conversion throws if an index falls outside hgcalMergeLayerClusters.
_extraLayerClusters = [tag if isinstance(tag, str) else tag.value() for tag in _hgcalMergeLayerClusters.extraLayerClusters]
assert not _extraLayerClusters or _extraLayerClusters[:len(_barrelLayerClusters)] == _barrelLayerClusters, \
    'hgcalMergeLayerClusters must merge the barrel layer clusters first among its extra layer clusters, in the order of _barrelComponents'

barrelSoAMergeLayerClusters = _hgCalSoAMergeLayerClustersProducer.clone(
    hgcalLayerClustersSoA = ['barrelSoALayerClusters' + component for component in _barrelComponents]
)

ticlCLUEsteringBarrel = _barrelCLUEsteringTrackstersProducer.clone(
    layerClustersSoA = 'barrelSoAMergeLayerClusters'
)

# The barrel PCA uses all the layer clusters: EB is a single layer, so the cleaning would keep one of them
ticlSoATrackstersBarrel = _hgCalSoATrackstersProducer.clone(
    hgcalLayerClustersSoA = 'barrelSoAMergeLayerClusters',
    hgcalLayerClustersTrackstersSoA = 'ticlCLUEsteringBarrel',
    detector = 'Barrel',
    usePCACleaning = False
)

# Conversion to legacy tracksters
ticlTrackstersCLUEsteringBarrel = _hgCalTrackstersFromSoAProducer.clone(
    src = 'ticlSoATrackstersBarrel',
    originalMask = 'hgcalMergeLayerClusters:InitialLayerClustersMask',
    precedingLayerClusters = _hgcalMergeLayerClusters.layerClustersSoA.value()
)

ticlCLUEsteringBarrelTask = cms.Task(barrelSoAMergeLayerClusters
    ,ticlCLUEsteringBarrel
    ,ticlSoATrackstersBarrel
    ,ticlTrackstersCLUEsteringBarrel)
