#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

EcalIntercalibConstantsGPU::EcalIntercalibConstantsGPU(EcalIntercalibConstants const& values) {
  std::cout << "check v1" << std::endl;
  values_.reserve(values.size());
  std::cout << "check v2" << std::endl;
  values_.insert(values_.end(), values.barrelItems().begin(), values.barrelItems().end());
  std::cout << "check v3" << std::endl;
  values_.insert(values_.end(), values.endcapItems().begin(), values.endcapItems().end());
  std::cout << "check v4" << std::endl;
  offset_ = values.barrelItems().size();
  std::cout << "check v5" << std::endl;
}

EcalIntercalibConstantsGPU::Product const& EcalIntercalibConstantsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](EcalIntercalibConstantsGPU::Product& product, cudaStream_t cudaStream) {
        // allocate
        product.values = cms::cuda::make_device_unique<float[]>(values_.size(), cudaStream);
        // transfer
        cms::cuda::copyAsync(product.values, values_, cudaStream);
      });

  return product;
}

TYPELOOKUP_DATA_REG(EcalIntercalibConstantsGPU);
