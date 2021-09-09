#include "Validation/HGCalValidation/plugins/HeterogeneousHGCalCLUEValidator.h"

HeterogeneousHGCalCLUEValidator::HeterogeneousHGCalCLUEValidator(const edm::ParameterSet &ps)
  : tokens_({{{{consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("cpuCLUEEMToken")),
		consumes<reco::BasicClusterCollection>(ps.getParameter<edm::InputTag>("gpuCLUEEMToken"))}},
      }}),
    treenames_({{"EM" /*, HAD */}}) {
  std::cout << "CONSTRUCTOR ========================" << std::endl;
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  for (unsigned i(0); i < nsubdetectors; ++i) {
    trees_[i] = fs->make<TTree>(treenames_[i].c_str(), treenames_[i].c_str());
    trees_[i]->Branch("cpu", "ValidCLUEClusterCollection", &cpuValidCLUEHits[i]);
    trees_[i]->Branch("gpu", "ValidCLUEClusterCollection", &gpuValidCLUEHits[i]);
    trees_[i]->Branch("diffs", "ValidCLUEClusterCollection", &diffsValidCLUEHits[i]);
  }
  std::cout << "CONSTRUCTOR END ========================" << std::endl;
}

HeterogeneousHGCalCLUEValidator::~HeterogeneousHGCalCLUEValidator() {}

void HeterogeneousHGCalCLUEValidator::endJob() {}

void HeterogeneousHGCalCLUEValidator::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  //future subdetector loop
  for (size_t idet = 0; idet < nsubdetectors; ++idet) {
    //get hits produced with the CPU
    const auto &cpuClusters = event.get(tokens_[idet][0]);

    //get hits produced with the GPU
    const auto &gpuClusters = event.get(tokens_[idet][1]);

    size_t nclusters = cpuClusters.size();
    std::cout << nclusters << ", " << gpuClusters.size() << std::endl;
    assert(nclusters == gpuClusters.size());
    //float sum_cpu = 0.f, sum_gpu = 0.f, sum_son_cpu = 0.f, sum_son_gpu = 0.f;
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

      cpuValidCLUEHits[idet].push_back(vCPU);
      gpuValidCLUEHits[idet].push_back(vGPU);
      diffsValidCLUEHits[idet].push_back(vDiffs);
    }
    std::cout << "CHECK" << std::endl;
    trees_[idet]->Fill();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeterogeneousHGCalCLUEValidator);
