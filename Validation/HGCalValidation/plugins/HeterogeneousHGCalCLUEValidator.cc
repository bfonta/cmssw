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

    size_t nclusters = cpuClusters.size();
    std::cout << nclusters << ", " << gpuClusters.size() << std::endl;

    //assers(nclusters == gpuClusters.size());
    //float sum_cpu = 0.f, sum_gpu = 0.f, sum_son_cpu = 0.f, sum_son_gpu = 0.f;

    //Clusters loop
    for (unsigned i(0); i < nclusters; i++) {
      const reco::BasicCluster &cpuCluster = cpuClusters[i];
      const reco::BasicCluster &gpuCluster = gpuClusters[i];

      const float cpuEn = cpuCluster.energy();
      const float gpuEn = gpuCluster.energy();

      const float cpuX = cpuCluster.x();
      const float gpuX = gpuCluster.x();
      const float cpuY = cpuCluster.y();
      const float gpuY = gpuCluster.y();
      const float cpuZ = cpuCluster.z();
      const float gpuZ = gpuCluster.z();

      ValidCLUECluster vCPU(cpuEn, cpuX, cpuY, cpuZ);
      ValidCLUECluster vGPU(gpuEn, gpuX, gpuY, gpuZ);
      ValidCLUECluster vDiffs(cpuEn - gpuEn,
			      cpuX - gpuX,
			      cpuY - gpuY,
			      cpuZ - gpuZ);

      cpuValidClusters[idet].push_back(vCPU);
      gpuValidClusters[idet].push_back(vGPU);
      diffsValidClusters[idet].push_back(vDiffs);
    }
    treesC_[idet]->Fill();

    //Hits loop
    size_t nlayers = cpuHits.size();
    std::cout << nlayers << std::endl;

    for (unsigned i(0); i<nlayers; i++) {
      const CellsOnLayer &cpuHitsOnLayer = cpuHits[i];

      for (unsigned j(0); j<cpuHitsOnLayer.detid.size(); j++) {
      
	const float cpuRho = cpuHitsOnLayer.rho[i];
	const float cpuDelta = cpuHitsOnLayer.delta[i];
	const float cpuNH = cpuHitsOnLayer.nearestHigher[i];
	const float cpuClusterIndex = cpuHitsOnLayer.clusterIndex[i];
	const float cpuId = cpuHitsOnLayer.detid[i];
	const float cpuIsSeed = cpuHitsOnLayer.isSeed[i];

	cpuValidHits[idet].emplace_back(cpuRho,
					cpuDelta,
					cpuNH,
					cpuClusterIndex,
					cpuId,
					cpuIsSeed);

      }
    }

    for (unsigned i(0); i<gpuProd.nHits(); i++) {
      const float gpuRho = gpuHits.rho[i];
      const float gpuDelta = gpuHits.delta[i];
      const float gpuNH = gpuHits.nearestHigher[i];
      const float gpuClusterIndex = gpuHits.clusterIndex[i];
      const float gpuId = gpuHits.id[i];
      const float gpuIsSeed = gpuHits.isSeed[i];

      gpuValidHits[idet].emplace_back(gpuRho,
				      gpuDelta,
				      gpuNH,
				      gpuClusterIndex,
				      gpuId,
				      gpuIsSeed);

    }
    
    diffsValidHits[idet].emplace_back(0.f,
				      0.f,
				      0,
				      0,
				      0,
				      false);
    treesH_[idet]->Fill();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeterogeneousHGCalCLUEValidator);
