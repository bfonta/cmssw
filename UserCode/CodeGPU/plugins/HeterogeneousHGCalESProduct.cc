#include "UserCode/CodeGPU/plugins/HeterogeneousHGCalESProduct.h"

HeterogeneousConditionsESProductWrapper::HeterogeneousConditionsESProductWrapper(unsigned int const& nelements, unsigned int const& stride, HeterogeneousConditionsESProduct const& cpuConditions):
  stride_(stride)
{
  chunk_ = 2*sizeof(int)*stride_;
  cudaCheck(cudaMallocHost(&this->layer_, chunk_));
  this->wafer_ = this->layer_ + sizeof(int)*stride_;
  for(unsigned int i=0; i<nelements; ++i)
    {
      this->layer_[i] = cpuConditions.layer[i];
      this->wafer_[i] = cpuConditions.wafer[i];
    }
}

HeterogeneousConditionsESProductWrapper::~HeterogeneousConditionsESProductWrapper() {
  cudaCheck(cudaFreeHost(this->layer_));
}

HeterogeneousConditionsESProduct const *HeterogeneousConditionsESProductWrapper::getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const {
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
	    cudaCheck(cudaMallocHost(&data.host, sizeof(HeterogeneousConditionsESProduct)));

	    // Allocate the payload array(s) on device memory.
	    cudaCheck(cudaMalloc(&data.host->layer, chunk_));
	    data.host->wafer = data.host->layer + sizeof(int)*stride_;

	    // Allocate the payload object on the device memory.
	    cudaCheck(cudaMalloc(&data.device, sizeof(HeterogeneousConditionsESProduct)));
	    	    
	    // Transfer the payload, first the array(s) ...
	    cudaCheck(cudaMemcpyAsync(data.host->layer, this->layer_, chunk_, cudaMemcpyHostToDevice, stream));
	    // ... and then the payload object
	    cudaCheck(cudaMemcpyAsync(data.device, data.host, sizeof(HeterogeneousConditionsESProduct), cudaMemcpyHostToDevice, stream));
	  }); //gpuData_.dataForCurrentDeviceAsync

  // Returns the payload object on the memory of the current device
  return data.device;
}

// Destructor frees all member pointers
HeterogeneousConditionsESProductWrapper::GPUData::~GPUData() {
  if(host != nullptr) 
    {
      cudaCheck(cudaFree(host->layer));
      cudaCheck(cudaFreeHost(host));
    }
  cudaCheck(cudaFree(device));
}
