import FWCore.ParameterSet.Config as cms

hgcal_layerClusters = [
    "hltHgCalLayerClustersFromSoAProducer",
    "hltHgCalLayerClustersFromSoAProducerHSci",
    "hltHgCalLayerClustersFromSoAProducerHSi"
]
hgcal_time_layerClusters = [x + ":timeLayerCluster" for x in hgcal_layerClusters]

hgcal_layerClustersSerialSync = [x + "SerialSync" for x in hgcal_layerClusters]
hgcal_time_layerClustersSerialSync = [x + "SerialSync:timeLayerCluster" for x in hgcal_layerClusters]

barrel_layerClusters = [
    "hltBarrelLayerClustersEB",
    "hltBarrelLayerClustersHB"
]
barrel_time_layerClusters = [x + ":timeLayerCluster" for x in barrel_layerClusters]

hltMergeLayerClusters = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag(*hgcal_layerClusters),
    time_layerclusters = cms.VInputTag(*hgcal_time_layerClusters),
)

hltMergeLayerClustersSerialSync = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag(*hgcal_layerClustersSerialSync),
    time_layerclusters = cms.VInputTag(*hgcal_time_layerClustersSerialSync),
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

ticl_barrel.toModify(hltMergeLayerClusters,
    layerClusters = [*hgcal_layerClusters, *barrel_layerClusters],
    time_layerclusters = [*hgcal_time_layerClusters, *barrel_time_layerClusters]
)

# The barrel has 'SerialSync' layer clusters only with its alpaka chain
barrel_layerClustersSerialSync = [x + "SerialSync" for x in barrel_layerClusters]
barrel_time_layerClustersSerialSync = [x + "SerialSync:timeLayerCluster" for x in barrel_layerClusters]

from Configuration.ProcessModifiers.alpaka_cff import alpaka
(ticl_barrel & alpaka).toModify(hltMergeLayerClustersSerialSync,
    layerClusters = [*hgcal_layerClustersSerialSync, *barrel_layerClustersSerialSync],
    time_layerclusters = [*hgcal_time_layerClustersSerialSync, *barrel_time_layerClustersSerialSync]
)
