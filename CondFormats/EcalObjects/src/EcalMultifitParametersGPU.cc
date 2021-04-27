#include "CondFormats/EcalObjects/interface/EcalMultifitParametersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalMultifitParametersGPU::EcalMultifitParametersGPU(edm::ParameterSet const& ps) {
  auto const& amplitudeFitParametersEB = ps.getParameter<std::vector<double>>("EBamplitudeFitParameters");
  auto const& amplitudeFitParametersEE = ps.getParameter<std::vector<double>>("EEamplitudeFitParameters");
  auto const& timeFitParametersEB = ps.getParameter<std::vector<double>>("EBtimeFitParameters");
  auto const& timeFitParametersEE = ps.getParameter<std::vector<double>>("EEtimeFitParameters");

  amplitudeFitParametersEB_.assign(amplitudeFitParametersEB.begin(),
				   amplitudeFitParametersEB.end());
  amplitudeFitParametersEE_.assign(amplitudeFitParametersEE.begin(),
				   amplitudeFitParametersEE.end());
  timeFitParametersEB_.assign(timeFitParametersEB.begin(),
			      timeFitParametersEB.end());
  timeFitParametersEE_.assign(timeFitParametersEE.begin(),
			      timeFitParametersEE.end());
}

EcalMultifitParametersGPU::Product const& EcalMultifitParametersGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalMultifitParametersGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.amplitudeFitParametersEB =
            cms::cuda::make_device_unique<double[]>(amplitudeFitParametersEB_.size(), cudaStream);
        product.amplitudeFitParametersEE =
            cms::cuda::make_device_unique<double[]>(amplitudeFitParametersEE_.size(), cudaStream);
        product.timeFitParametersEB = cms::cuda::make_device_unique<double[]>(timeFitParametersEB_.size(), cudaStream);
        product.timeFitParametersEE = cms::cuda::make_device_unique<double[]>(timeFitParametersEE_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.amplitudeFitParametersEB, amplitudeFitParametersEB_, cudaStream);
        cms::cuda::copyAsync(product.amplitudeFitParametersEE, amplitudeFitParametersEE_, cudaStream);
        cms::cuda::copyAsync(product.timeFitParametersEB, timeFitParametersEB_, cudaStream);
        cms::cuda::copyAsync(product.timeFitParametersEE, timeFitParametersEE_, cudaStream);
      });
  return product;
}

TYPELOOKUP_DATA_REG(EcalMultifitParametersGPU);
