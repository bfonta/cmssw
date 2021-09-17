#include "Validation/HGCalValidation/plugins/HeterogeneousHGCalCLUEValidator.h"

HeterogeneousHGCalCLUEValidator::HeterogeneousHGCalCLUEValidator(const edm::ParameterSet &ps)
  : tokHitsCPU_({{consumes<InHitsCPU>(ps.getParameter<edm::InputTag>("cpuHitsEMToken"))}}),
    tokHitsGPU_({{consumes<InHitsGPU>(ps.getParameter<edm::InputTag>("gpuHitsEMToken"))}}),
    tokClustersCPU_({{consumes<InClustersCPU>(ps.getParameter<edm::InputTag>("cpuClustersEMToken"))}}),
    tokClustersGPU_({{consumes<InClustersGPU>(ps.getParameter<edm::InputTag>("gpuClustersEMToken"))}}),
    treenamesH_({{"HitsEM" /*, HitsEM */}}),
    treenamesC_({{"ClustersEM" /*, ClustersHAD */}}) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  for (unsigned i(0); i < nReg; ++i) {
    treesC_[i] = fs->make<TTree>(treenamesC_[i].c_str(), treenamesC_[i].c_str());
    treesC_[i]->Branch("cpu", "ValidCLUEClusterCollection", &cpuValidClusters[i]);
    treesC_[i]->Branch("gpu", "ValidCLUEClusterCollection", &gpuValidClusters[i]);
    treesC_[i]->Branch("diffs", "ValidCLUEClusterCollection", &diffsValidClusters[i]);

    treesH_[i] = fs->make<TTree>(treenamesH_[i].c_str(), treenamesH_[i].c_str());
    treesH_[i]->Branch("cpu", "ValidCLUEHitCollection", &cpuValidHits[i]);
    treesH_[i]->Branch("gpu", "ValidCLUEHitCollection", &gpuValidHits[i]);
    treesH_[i]->Branch("diffs", "ValidCLUEHitCollection", &diffsValidHits[i]);
  }
}

HeterogeneousHGCalCLUEValidator::~HeterogeneousHGCalCLUEValidator() {}

void HeterogeneousHGCalCLUEValidator::endJob() {}

void HeterogeneousHGCalCLUEValidator::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  //future subdetector loop
  for (size_t idet = 0; idet < nReg; ++idet) {
    const auto &cpuHits = event.get(tokHitsCPU_[idet]);
    const auto &cpuClusters = event.get(tokClustersCPU_[idet]);

    const auto &gpuProd = event.get(tokHitsGPU_[idet]);
    ConstHGCCLUEHitsSoA gpuHits = gpuProd.get();
    const auto &gpuClusters = event.get(tokClustersGPU_[idet]);

    size_t nClustersCPU = cpuClusters.size();
    size_t nClustersGPU = gpuClusters.size();
    std::cout << nClustersCPU << ", " << nClustersGPU << std::endl;

    //assers(nclusters == gpuClusters.size());
    //float sum_cpu = 0.f, sum_gpu = 0.f, sum_son_cpu = 0.f, sum_son_gpu = 0.f;

    //CPU clusters loop
    for (unsigned i(0); i < nClustersCPU; i++) {
      const reco::BasicCluster &cpuCluster = cpuClusters[i];

      if(cpuCluster.algo() == reco::CaloCluster::hgcal_em)
	{  
	  const float cpuEn = cpuCluster.energy();
	  const float cpuX = cpuCluster.x();
	  const float cpuY = cpuCluster.y();
	  const float cpuZ = cpuCluster.z();

	  ValidCLUECluster vCPU(cpuEn, cpuX, cpuY, cpuZ);

	  cpuValidClusters[idet].emplace_back(cpuEn, cpuX, cpuY, cpuZ);
	}
      else if(cpuCluster.algo() != reco::CaloCluster::hgcal_em)
	std::cout << "cpu Cluster ERROR: " << cpuCluster.algo() << std::endl;
    }
    
    //GPU clusters loop
    for (unsigned i(0); i < nClustersGPU; i++) {
      const reco::BasicCluster &gpuCluster = gpuClusters[i];

      const float gpuEn = gpuCluster.energy();
      const float gpuX = gpuCluster.x();
      const float gpuY = gpuCluster.y();
      const float gpuZ = gpuCluster.z();

      ValidCLUECluster vGPU(gpuEn, gpuX, gpuY, gpuZ);

      gpuValidClusters[idet].emplace_back(gpuEn, gpuX, gpuY, gpuZ);
    }
    treesC_[idet]->Fill();

    //Hits loop
    size_t nlayers = cpuHits.size();
    std::cout << "NLAYERS: " << nlayers << std::endl;

    for (unsigned i(0); i<nlayers; i++) {
      const CellsOnLayer &cpuHitsOnLayer = cpuHits[i];

      for (unsigned j(0); j<cpuHitsOnLayer.detid.size(); j++) {

	const float cpuId = cpuHitsOnLayer.detid[i];

	if(DetId(cpuId).det()==DetId::HGCalEE or DetId(cpuId).det()==DetId::HGCalHSi)
	  {
	    const float cpuRho = cpuHitsOnLayer.rho[j];
	    const float cpuDelta = cpuHitsOnLayer.delta[j];
	    const float cpuNH = cpuHitsOnLayer.nearestHigher[j];
	    const float cpuClusterIndex = cpuHitsOnLayer.clusterIndex[j];
	    const float cpuLayer = i;
	    const float cpuIsSeed = cpuHitsOnLayer.isSeed[j];

	    cpuValidHits[idet].emplace_back(cpuRho,
					    cpuDelta,
					    cpuNH,
					    cpuClusterIndex,
					    cpuLayer,
					    cpuId,
					    cpuIsSeed);

	  }
	else if(DetId(cpuId).det()!=DetId::HGCalHSc)
	  std::cout << "cpu Hits ERROR: " << DetId(cpuId).det() << std::endl;
      }
    }

    for (unsigned i(0); i<gpuProd.nHits(); i++) {
      const float gpuRho = gpuHits.rho[i];
      const float gpuDelta = gpuHits.delta[i];
      const int32_t gpuNH = gpuHits.nearestHigher[i];
      const int32_t gpuClusterIndex = gpuHits.clusterIndex[i];
      const uint32_t gpuId = gpuHits.id[i];
      HGCSiliconDetId thisId(gpuId); //HGCalDetId.h is obsolete and should be removed!!!
      const int32_t gpuLayer = thisId.zside()*thisId.layer();
      const bool gpuIsSeed = gpuHits.isSeed[i];

      gpuValidHits[idet].emplace_back(gpuRho,
				      gpuDelta,
				      gpuNH,
				      gpuClusterIndex,
				      gpuLayer,
				      gpuId,
				      gpuIsSeed);

    }
    
    diffsValidHits[idet].emplace_back(0.f,
				      0.f,
				      0,
				      0,
				      0,
				      0,
				      false);
    treesH_[idet]->Fill();
  }
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeterogeneousHGCalCLUEValidator);
