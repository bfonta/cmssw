#include <iostream>
#include <memory>
#include <chrono>
#include <numeric>
#include <string>
#include <fstream>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitCPUProduct.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitCPUProduct.h"
#include "CUDADataFormats/HGCal/interface/ConstHGCRecHitSoA.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"

#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitCPUProduct.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/MessageLogger.h"

#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalRecHit.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitGPUProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitDevice.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibRecHitHost.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/MessageDefinition.pb.h"

class EERecHitFull : public edm::stream::EDProducer<> {
public:
  explicit EERecHitFull(const edm::ParameterSet& ps);
  ~EERecHitFull() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  void convert_soa_data_to_collection_(uint32_t, HGCRecHitCollection&, ConstHGCRecHitSoA*);

private:
  std::vector<double> totaltime;
  
  edm::EDGetTokenT<HGCeeUncalibratedRecHitCollection> uncalibRecHitCPUToken_;
  edm::EDPutTokenT<HGCeeRecHitCollection> recHitCollectionToken_;

  HGCeeUncalibRecHitConstantData cdata_;
  HGCConstantVectorData vdata_;

  std::string assert_error_message_(std::string, const size_t &, const size_t &);
  void assert_sizes_constants_(const HGCConstantVectorData &);

  std::unique_ptr<hgcal::RecHitTools> tools_;

  void convert_collection_data_to_soa_(const uint32_t &,
                                       const HGCeeUncalibratedRecHitCollection &);
  void convert_constant_data_(KernelConstantData<HGCeeUncalibRecHitConstantData> *);

  HGCRecHitGPUProduct prodGPU_;
  HGCRecHitCPUProduct prodCPU_;
  HGCUncalibRecHitDevice d_uncalib_;
  //HGCUncalibRecHitHost<HGCeeUncalibratedRecHitCollection> h_uncalib_;
  HGCUncalibRecHitHost<uncalibRecHitsProtocol::Event> h_uncalib_;
  std::unique_ptr<HGCeeRecHitCollection> rechits_;
  KernelConstantData<HGCeeUncalibRecHitConstantData> *kcdata_;

  uncalibRecHitsProtocol::Data binary_data_;
  std::string fileName_;
  unsigned nEvents_;
  unsigned counter_;
};

EERecHitFull::EERecHitFull(const edm::ParameterSet& ps):
  uncalibRecHitCPUToken_{consumes<HGCUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("HGCEEUncalibRecHitsTok"))},
  fileName_{ps.getParameter<std::string>("fileName")},
  nEvents_{ps.getParameter<unsigned>("nEvents")}
{
  recHitCollectionToken_ = produces<HGCeeRecHitCollection>();
  
  cdata_.keV2DIGI_ = ps.getParameter<double>("HGCEE_keV2DIGI");
  cdata_.xmin_ = ps.getParameter<double>("minValSiPar");  //float
  cdata_.xmax_ = ps.getParameter<double>("maxValSiPar");  //float
  cdata_.aterm_ = ps.getParameter<double>("noiseSiPar");  //float
  cdata_.cterm_ = ps.getParameter<double>("constSiPar");  //float
  vdata_.fCPerMIP_ = ps.getParameter<std::vector<double>>("HGCEE_fCPerMIP");
  vdata_.cce_ = ps.getParameter<edm::ParameterSet>("HGCEE_cce").getParameter<std::vector<double>>("values");
  vdata_.noise_fC_ = ps.getParameter<edm::ParameterSet>("HGCEE_noise_fC").getParameter<std::vector<double>>("values");
  vdata_.rcorr_ = ps.getParameter<std::vector<double>>("rcorr");
  vdata_.weights_ = ps.getParameter<std::vector<double>>("weights");
  cdata_.uncalib2GeV_ = 1e-6 / cdata_.keV2DIGI_;
  assert_sizes_constants_(vdata_);

  kcdata_ = new KernelConstantData<HGCeeUncalibRecHitConstantData>(cdata_, vdata_);
  convert_constant_data_(kcdata_);
  
  tools_ = std::make_unique<hgcal::RecHitTools>();

  //Protocol Buffer initialization
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  std::ifstream input(fileName_, std::ios::in);
  if (!binary_data_.ParseFromIstream(&input)) {
    edm::LogError("ParseError") << "Failed to parse.";
  }
  counter_ = 0;
}

EERecHitFull::~EERecHitFull() {
  delete kcdata_;
  /*
  for(unsigned i(0); i<totaltime.size(); ++i)
    std::cout << totaltime[i] << ", ";
  double sum = std::accumulate(totaltime.begin(), totaltime.end(), 0.);
  unsigned tsize = totaltime.size();
  double mean = sum / static_cast<double>(tsize);
  double sq_sum = std::inner_product(totaltime.begin(), totaltime.end(), totaltime.begin(), 0.0);
  double stdev = std::sqrt(sq_sum/tsize - mean*mean);
  std::cout << "TOTAL GPU " <<  sum << std::endl;
  std::cout << "mean " << mean << std::endl;
  std::cout << "std " << stdev << std::endl;
  */
  google::protobuf::ShutdownProtobufLibrary();
}

std::string EERecHitFull::assert_error_message_(std::string var, const size_t& s1, const size_t& s2) {
  std::string str1 = "The '";
  std::string str2 = "' array must be at least of size ";
  std::string str3 = " to hold the configuration data, but is of size ";
  return str1 + var + str2 + std::to_string(s1) + str3 + std::to_string(s2);
}

void EERecHitFull::assert_sizes_constants_(const HGCConstantVectorData& vd) {
  if (vdata_.fCPerMIP_.size() < HGCeeUncalibRecHitConstantData::ee_fCPerMIP)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "fCPerMIP", HGCeeUncalibRecHitConstantData::ee_fCPerMIP, vdata_.fCPerMIP_.size());
  else if (vdata_.cce_.size() < HGCeeUncalibRecHitConstantData::ee_cce)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "cce", HGCeeUncalibRecHitConstantData::ee_cce, vdata_.cce_.size());
  else if (vdata_.noise_fC_.size() < HGCeeUncalibRecHitConstantData::ee_noise_fC)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "noise_fC", HGCeeUncalibRecHitConstantData::ee_noise_fC, vdata_.noise_fC_.size());
  else if (vdata_.rcorr_.size() < HGCeeUncalibRecHitConstantData::ee_rcorr)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "rcorr", HGCeeUncalibRecHitConstantData::ee_rcorr, vdata_.rcorr_.size());
  else if (vdata_.weights_.size() < HGCeeUncalibRecHitConstantData::ee_weights)
    edm::LogError("WrongSize") << this->assert_error_message_(
        "weights", HGCeeUncalibRecHitConstantData::ee_weights, vdata_.weights_.size());
}

void EERecHitFull::produce(edm::Event& event, const edm::EventSetup& setup) {  
  cms::cuda::ScopedContextProduce ctx{event.streamID()};

  ++counter_;
  const uncalibRecHitsProtocol::Event& hits = binary_data_.events( counter_ % nEvents_ );
  const unsigned nhits(hits.amplitude_size());

  /*
  const auto& hits = event.get(uncalibRecHitCPUToken_);
  const unsigned nhits(hits.size());
  */

  if (nhits == 0)
    cms::cuda::LogError("EERecHitFull") << "WARNING: no input hits!";

  auto start = std::chrono::high_resolution_clock::now();
  
  prodGPU_   = HGCRecHitGPUProduct(nhits, ctx.stream());
  d_uncalib_ = HGCUncalibRecHitDevice(nhits, ctx.stream());
  //h_uncalib_ = HGCUncalibRecHitHost<HGCeeUncalibratedRecHitCollection>(nhits, hits, ctx.stream());
  h_uncalib_ = HGCUncalibRecHitHost<uncalibRecHitsProtocol::Event>(nhits, hits, ctx.stream());

  KernelManagerHGCalRecHit km1(h_uncalib_.get(), d_uncalib_.get(), prodGPU_.get());
  km1.run_kernels(kcdata_, ctx.stream());
  //add CUDA device synchronize

  auto finish = std::chrono::high_resolution_clock::now();
    
  prodCPU_ = HGCRecHitCPUProduct(prodGPU_.nHits(), ctx.stream());
  KernelManagerHGCalRecHit km2(prodCPU_.get(), prodGPU_.get());
  km2.transfer_soa_to_host(ctx.stream());
  //add CUDA device synchronize
    
  rechits_ = std::make_unique<HGCRecHitCollection>();
  ConstHGCRecHitSoA tmpSoA = prodCPU_.getConst();
  convert_soa_data_to_collection_(prodCPU_.getConst().nhits_, *rechits_, &tmpSoA);

  std::chrono::duration<double> elapsed = finish - start;
  //std::cout << "GPU " << elapsed.count()*1000  << std::endl;
  totaltime.push_back( elapsed.count()*1000 );
  
  event.put(std::move(rechits_));
}

void EERecHitFull::convert_soa_data_to_collection_(uint32_t nhits,
						   HGCRecHitCollection& rechits,
						   ConstHGCRecHitSoA* h_calibSoA) {
  rechits.reserve(nhits);
  for (uint i = 0; i < nhits; ++i) {
    DetId id_converted(h_calibSoA->id_[i]);
    rechits.emplace_back(id_converted,
			 h_calibSoA->energy_[i],
			 h_calibSoA->time_[i],
			 0,
			 h_calibSoA->flagBits_[i],
			 h_calibSoA->son_[i],
			 h_calibSoA->timeError_[i]);
  }
}

void EERecHitFull::convert_constant_data_(KernelConstantData<HGCeeUncalibRecHitConstantData>* kcdata) {
  for (size_t i = 0; i < kcdata->vdata_.fCPerMIP_.size(); ++i)
    kcdata->data_.fCPerMIP_[i] = kcdata->vdata_.fCPerMIP_[i];
  for (size_t i = 0; i < kcdata->vdata_.cce_.size(); ++i)
    kcdata->data_.cce_[i] = kcdata->vdata_.cce_[i];
  for (size_t i = 0; i < kcdata->vdata_.noise_fC_.size(); ++i)
    kcdata->data_.noise_fC_[i] = kcdata->vdata_.noise_fC_[i];
  for (size_t i = 0; i < kcdata->vdata_.rcorr_.size(); ++i)
    kcdata->data_.rcorr_[i] = kcdata->vdata_.rcorr_[i];
  for (size_t i = 0; i < kcdata->vdata_.weights_.size(); ++i)
    kcdata->data_.weights_[i] = kcdata->vdata_.weights_[i];
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EERecHitFull);
