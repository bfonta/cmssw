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

  for(unsigned i = 0; i<nTechnologies; ++i) {
    std::string s = (i==0) ? "_ClusterCPU" : "_ClusterGPU";
    histosEn[i] = fs->make<TH1F>(("Energy" + s).c_str(),    "En", 80,  0., 1. );
    histosX[i]  = fs->make<TH1F>(("PositionX" + s).c_str(), "X", 200,  -160, 160 );
    histosY[i]  = fs->make<TH1F>(("PositionY" + s).c_str(), "Y", 200,  -160, 160 );
    histosZ[i]  = fs->make<TH1F>(("PositionZ" + s).c_str(), "Z", 4000,  -400, 400 );
  }

  fileCPU.open("ids_CPU.txt", std::ios_base::out);
  fileGPU.open("ids_GPU.txt", std::ios_base::out);
}

HeterogeneousHGCalCLUEValidator::~HeterogeneousHGCalCLUEValidator() {
  fileCPU.close();
  fileGPU.close();
}

void HeterogeneousHGCalCLUEValidator::endJob() {}

const reco::BasicCluster HeterogeneousHGCalCLUEValidator::get_matched_cluster(
				      unsigned nClustersGPU,
				      uint32_t cpuSeedId,
				      const std::vector<reco::BasicCluster>& gpuClusters) {
  for (unsigned i(0); i<nClustersGPU; i++) {
    const reco::BasicCluster &gpuCluster = gpuClusters[i];
    if(cpuSeedId == gpuCluster.seed())
      return gpuCluster;
  }

  //return dummy cluster if it did not find a match
  std::vector<std::pair<DetId, float>> dummy;
  return reco::BasicCluster(-1.,
			    math::XYZPoint(0.f, 0.f, 0.f),
			    reco::CaloID::DET_HGCAL_ENDCAP,
			    dummy,
			    reco::CaloCluster::undefined);
}

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
    std::cout << "[CLUEValidator] #Clusters: CPU=" << nClustersCPU
	      << ", GPU=" << nClustersGPU << std::endl;

    ///////////////////////////////////////////////
    //CPU clusters loop ///////////////////////////
    ///////////////////////////////////////////////
    for (unsigned i(0); i < nClustersCPU; i++) {
      const reco::BasicCluster &cpuCluster = cpuClusters[i];

      //print detids to files as an extra check
      if(i==0) {
	for (unsigned i(0); i < nClustersCPU; i++) {
	  if (!fileCPU.is_open())
	    std::cout << "failed to open CPU file" << std::endl;
	  else
	    fileCPU << i << ": " << cpuClusters[i].seed().rawId() << std::endl;
	}
      }

      if(cpuCluster.algo() == reco::CaloCluster::hgcal_em)
	{  
	  const float cpuEn = cpuCluster.energy();
	  histosEn[0]->Fill(cpuEn);
	  const float cpuX = cpuCluster.x();
	  histosX[0]->Fill(cpuX);
	  const float cpuY = cpuCluster.y();
	  histosY[0]->Fill(cpuY);
	  const float cpuZ = cpuCluster.z();
	  histosZ[0]->Fill(cpuZ);

	  cpuValidClusters[idet].emplace_back(cpuEn, cpuX, cpuY, cpuZ);
	}
      else if(cpuCluster.algo() != reco::CaloCluster::hgcal_em)
	std::cout << "cpu Cluster ERROR: " << cpuCluster.algo() << std::endl;
    }
    
    //GPU clusters loop
    for (unsigned i(0); i < nClustersGPU; i++) {
      const reco::BasicCluster &gpuCluster = gpuClusters[i];

      //print detids to files as an extra check
      if(i==0) {
	for (unsigned i(0); i < nClustersGPU; i++) {
	  if (!fileGPU.is_open())
	    std::cout << "failed to open GPU file" << std::endl;
	  else
	    fileGPU << i << ": " << gpuClusters[i].seed().rawId() << std::endl;
	}
      }

      const float gpuEn = gpuCluster.energy();
      histosEn[1]->Fill(gpuEn);
      const float gpuX = gpuCluster.x();
      histosX[1]->Fill(gpuX);
      const float gpuY = gpuCluster.y();
      histosY[1]->Fill(gpuY);
      const float gpuZ = gpuCluster.z();
      histosZ[1]->Fill(gpuZ);
      
      ValidCLUECluster vGPU(gpuEn, gpuX, gpuY, gpuZ);

      gpuValidClusters[idet].emplace_back(gpuEn, gpuX, gpuY, gpuZ);
    }

    //search for a GPU cluster initiated by the same CPU seed
    unsigned counter = 0;
    for (unsigned i(0); i < nClustersCPU; i++) {
      const reco::BasicCluster &cpuCluster = cpuClusters[i];
      
      if(cpuCluster.algo() == reco::CaloCluster::hgcal_em)
	{  
	  const float cpuEn = cpuCluster.energy();
	  const float cpuX = cpuCluster.x();
	  const float cpuY = cpuCluster.y();
	  const float cpuZ = cpuCluster.z();
	  const uint32_t cpuSeedId = cpuCluster.seed().rawId();

	  const reco::BasicCluster &gpuCluster = get_matched_cluster(nClustersGPU,
								     cpuSeedId,
								     gpuClusters);

	  if(gpuCluster.algo() == reco::CaloCluster::undefined)
	    ++counter;

	  diffsValidClusters[idet].emplace_back( gpuCluster.energy() - cpuEn,
						 gpuCluster.x() - cpuX,
						 gpuCluster.y() - cpuY,
						 gpuCluster.z() - cpuZ );
	  
	  continue;
	}
    }
    std::cout << "RATIO: "  << counter << ", " << nClustersCPU << ", " << static_cast<float>(counter)/nClustersCPU << std::endl;
    treesC_[idet]->Fill();
    
    ///////////////////////////////////////////////
    //Hits loop ///////////////////////////////////
    ///////////////////////////////////////////////
    size_t nlayers = cpuHits.size();
    std::cout << "NLAYERS: " << nlayers << std::endl;

    for (unsigned i(0); i<nlayers; i++) {
      const CellsOnLayer &cpuHitsOnLayer = cpuHits[i];

      for (unsigned j(0); j<cpuHitsOnLayer.detid.size(); j++) {

	const DetId cpuId = cpuHitsOnLayer.detid[j];
	const float cpuRho = cpuHitsOnLayer.rho[j];
	const float cpuX = cpuHitsOnLayer.x[j];
	const float cpuY = cpuHitsOnLayer.y[j];
	const float cpuDelta = cpuHitsOnLayer.delta[j];
	const int32_t cpuNH = cpuHitsOnLayer.nearestHigher[j];
	const int32_t cpuClusterIndex = cpuHitsOnLayer.clusterIndex[j];
	const int32_t cpuLayer = i;
	const bool cpuIsSeed = cpuHitsOnLayer.isSeed[j];

	if(cpuId.det()==DetId::HGCalEE or cpuId.det()==DetId::HGCalHSi)
	  {
	    cpuValidHits[idet].emplace_back(cpuRho,
					    cpuDelta,
					    cpuX,
					    cpuY,
					    cpuNH,
					    cpuClusterIndex,
					    cpuLayer,
					    cpuId,
					    cpuIsSeed);

	  }
	else if(cpuId.det()!=DetId::HGCalHSc)
	  std::cout << "cpu Hits ERROR: " << cpuId.det() << std::endl;

	//matching with GPU hits
	bool found_match = false;
	for (unsigned k(0); k<gpuProd.nHits(); k++) {

	  if(cpuId.rawId() == gpuHits.id[k]) {
	    if(found_match == true)
	      std::cout << "Duplication ERROR: Hit " << cpuId.rawId()
			<< " found multiple times in the GPU!" << std::endl;
	    
	    found_match = true;
	
	    //filter only "good" hits for comparison
	    if(gpuHits.nearestHigher[k] > -1 and gpuHits.clusterIndex[k] > -1) {
	      
	      diffsValidHits[idet].emplace_back(cpuRho - gpuHits.rho[k],
						cpuDelta - gpuHits.delta[k],
						cpuX - gpuHits.x[k],
						cpuY - gpuHits.y[k],
						0,
						0,
						0,
						0,
						cpuIsSeed == gpuHits.isSeed[k]);


	      if(cpuNH <= -1 or cpuClusterIndex <= -1) {
	      	std::cout << "ERROR: Quality " << cpuNH << ", " << cpuClusterIndex << ", "
	      		  << gpuHits.nearestHigher[k] << ", " << gpuHits.clusterIndex[k] << " :: "
	      		  << cpuRho << " :: " << gpuHits.rho[k]
	      		  << " Ids: " << gpuHits.id[k] << ", " << cpuId.rawId()
	      		  << std::endl;
	      }
	      else {
	      	if(std::abs(cpuDelta-gpuHits.delta[k])>0.01) {
	      	  std::cout << cpuRho << ", " << gpuHits.rho[k] << " :: "
	      		    << cpuNH << ", " << gpuHits.nearestHigher[k] << " :: "
	      		    << cpuClusterIndex << ", " << gpuHits.clusterIndex[k] << std::endl;
	      	}
	      }
	    }
	  }
	}

	if(found_match == false)
	  std::cout << "ERROR: Hit " << cpuId.rawId() << " not found in the GPU!" << std::endl;
      }
    }

    for (unsigned i(0); i<gpuProd.nHits(); i++) {
      const float gpuRho = gpuHits.rho[i];
      const float gpuDelta = gpuHits.delta[i];
      const float gpuX = gpuHits.x[i];
      const float gpuY = gpuHits.y[i];
      const int32_t gpuNH = gpuHits.nearestHigher[i];
      const int32_t gpuClusterIndex = gpuHits.clusterIndex[i];
      const uint32_t gpuId = gpuHits.id[i];
      HGCSiliconDetId thisId(gpuId); //HGCalDetId.h is obsolete and should be removed!!!
      const int32_t gpuLayer = thisId.zside()*thisId.layer();
      const bool gpuIsSeed = gpuHits.isSeed[i];

      gpuValidHits[idet].emplace_back(gpuRho,
				      gpuDelta,
				      gpuX,
				      gpuY,
				      gpuNH,
				      gpuClusterIndex,
				      gpuLayer,
				      gpuId,
				      gpuIsSeed);

    }
    
    treesH_[idet]->Fill();
  }
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeterogeneousHGCalCLUEValidator);
