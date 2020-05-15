#include "UserCode/CodeGPU/plugins/HeterogeneousHGCalESProduct.h"

HeterogeneousConditionsESProductWrapper::HeterogeneousConditionsESProductWrapper(unsigned int const& dddSize, const HGCalDDDConstants *cpuConditions)
{
  chunk1_ = dddSize;
  gpuErrchk(cudaMallocHost(&this->ddd_.waferTypeL, chunk1_));
  for(unsigned int i=0; i<dddSize; ++i)
    this->ddd_.waferTypeL[i] = cpuConditions->getParameter()->waferTypeL_[i];
}

HeterogeneousConditionsESProductWrapper::~HeterogeneousConditionsESProductWrapper() {
  gpuErrchk(cudaFreeHost(this->ddd_.waferTypeL));
}

HeterogeneousHEFConditionsESProduct const *HeterogeneousConditionsESProductWrapper::getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const {
  // cms::cuda::ESProduct<T> essentially holds an array of GPUData objects,
  // one per device. If the data have already been transferred to the
  // current device (or the transfer has been queued), the helper just
  // returns a reference to that GPUData object. Otherwise, i.e. data are
  // not yet on the current device, the helper calls the lambda to do the
  // necessary memory allocations and to queue the transfers.
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream,
	  [this](GPUData& data, cudaStream_t stream)
	  {    
	    // Allocate the payload object on pinned host memory.
	    gpuErrchk(cudaMallocHost(&data.host, sizeof(HeterogeneousHEFConditionsESProduct)));
	    // Allocate the payload array(s) on device memory.
	    gpuErrchk(cudaMalloc(&(data.host->ddd.waferTypeL), chunk1_));

	    // Allocate the payload object on the device memory.
	    gpuErrchk(cudaMalloc(&data.device, sizeof(HeterogeneousHEFConditionsESProduct)));
	    // Transfer the payload, first the array(s) ...
	    gpuErrchk(cudaMemcpyAsync(data.host->ddd.waferTypeL, this->ddd_.waferTypeL, chunk1_, cudaMemcpyHostToDevice, stream));
	    // ... and then the payload object
	    gpuErrchk(cudaMemcpyAsync(data.device, data.host, sizeof(HeterogeneousHEFConditionsESProduct), cudaMemcpyHostToDevice, stream));
	  }); //gpuData_.dataForCurrentDeviceAsync

  // Returns the payload object on the memory of the current device
  return data.device;
}

// Destructor frees all member pointers
HeterogeneousConditionsESProductWrapper::GPUData::~GPUData() {
  if(host != nullptr) 
    {
      gpuErrchk(cudaFree(host->ddd.waferTypeL));
      gpuErrchk(cudaFreeHost(host));
    }
  gpuErrchk(cudaFree(device));
}
