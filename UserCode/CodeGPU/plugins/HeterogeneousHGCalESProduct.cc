#include "UserCode/CodeGPU/plugins/HeterogeneousHGCalESProduct.h"

HeterogeneousConditionsESProductWrapper::HeterogeneousConditionsESProductWrapper(const HGCalParameters* cpuHGCalParameters)
{
  calculate_memory_bytes(cpuHGCalParameters);

  chunk_ = std::accumulate(this->sizes_.begin(), this->sizes_.end(), 0); //total memory required in bytes
  std::cout << "CHUNK: " << chunk_ << std::endl;
  gpuErrchk(cudaMallocHost(&this->params_.cellFineX_, chunk_));

  //store cumulative sum in bytes and convert it to sizes in units of C++ types, i.e., number if items to be transferred to GPU
  std::vector<size_t> cumsum_sizes( this->sizes_.size()+1, 0 ); //starting with zero
  std::partial_sum(this->sizes_.begin(), this->sizes_.end(), cumsum_sizes.begin()+1);
  for(unsigned int i=1; i<cumsum_sizes.size(); ++i) //start at second element (the first is zero)
    {
      unsigned int typesize;
      if( cp::types[i-1] == cp::HeterogeneousHGCalParametersType::Double )
	typesize = sizeof(double);
      else if( cp::types[i-1] == cp::HeterogeneousHGCalParametersType::Int32_t )
	typesize = sizeof(int32_t);
      else
	throw std::runtime_error("HeterogeneousConditionsESProductWrapper::HeterogeneousConditionsESProductWrapper(): wrong type.");
      cumsum_sizes[i] /= typesize;
    }

  for(unsigned int j=0; j<this->sizes_.size(); ++j) { 

    //setting the pointers
    if(j != 0)
      {
	const unsigned int jm1 = j-1;
	if( cp::types[jm1] == cp::HeterogeneousHGCalParametersType::Double and 
	    cp::types[j] == cp::HeterogeneousHGCalParametersType::Double )
	  select_pointer_d(&this->params_, j) = select_pointer_d(&this->params_, jm1) + this->sizes_[jm1];
	else if( cp::types[jm1] == cp::HeterogeneousHGCalParametersType::Double and 
		 cp::types[j] == cp::HeterogeneousHGCalParametersType::Int32_t )
	  select_pointer_i(&this->params_, j) = reinterpret_cast<int32_t*>( select_pointer_d(&this->params_, jm1) + this->sizes_[jm1] );
      }

    //copying the pointers' content
    for(unsigned int i=cumsum_sizes[j]; i<cumsum_sizes[j+1]; ++i) 
      {
	unsigned int index = i - cumsum_sizes[j];
	if( cp::types[j] == cp::HeterogeneousHGCalParametersType::Double ) {
	  select_pointer_d(&this->params_, j)[index] = select_pointer_d(cpuHGCalParameters, j)[index];
	}	  
	else if( cp::types[j] == cp::HeterogeneousHGCalParametersType::Int32_t )
	  select_pointer_i(&this->params_, j)[index] = select_pointer_i(cpuHGCalParameters, j)[index];
	else
	  throw std::runtime_error("HeterogeneousConditionsESProductWrapper::HeterogeneousConditionsESProductWrapper(): wrong type.");
      }
  }
}

void HeterogeneousConditionsESProductWrapper::calculate_memory_bytes(const HGCalParameters* cpuHGCalParameters) {
  size_t npointers = hgcal_conditions::parameters::types.size();
  std::vector<size_t> sizes(npointers);
  for(unsigned int i=0; i<npointers; ++i)
    {
      if(cp::types[i] == cp::HeterogeneousHGCalParametersType::Double)
	sizes[i] = select_pointer_d(cpuHGCalParameters, i).size();
      else
	sizes[i] = select_pointer_i(cpuHGCalParameters, i).size();
    }

  std::vector<size_t> sizes_units(npointers);
  for(unsigned int i=0; i<npointers; ++i)
    {
      if(cp::types[i] == cp::HeterogeneousHGCalParametersType::Double)
	sizes_units[i] = sizeof(double);
      else if(cp::types[i] == cp::HeterogeneousHGCalParametersType::Int32_t)
	sizes_units[i] = sizeof(int32_t);
    }

  for(unsigned int i=0; i<npointers; ++i) 
    {
      std::cout << sizes[i] << ", " << sizes_units[i] << std::endl;
    }

  //element by element multiplication
  this->sizes_.resize(npointers);
  std::transform( sizes.begin(), sizes.end(), sizes_units.begin(), this->sizes_.begin(), std::multiplies<size_t>() );
}

HeterogeneousConditionsESProductWrapper::~HeterogeneousConditionsESProductWrapper() {
  gpuErrchk(cudaFreeHost(this->params_.cellFineX_));
}

//I could use template specializations
//try to use std::variant in the future to avoid similar functions with different return values
double*& HeterogeneousConditionsESProductWrapper::select_pointer_d(cp::HeterogeneousHGCalParameters* cpuObject, 
								   const unsigned int& item) const {
  switch(item) 
    {
    case 0:
      return cpuObject->cellFineX_;
    case 1:
      return cpuObject->cellFineY_;
    case 2:
      return cpuObject->cellCoarseX_;
    case 3:
      return cpuObject->cellCoarseY_;
    default:
      throw std::runtime_error("HeterogeneousConditionsESProductWrapper::select_pointer_d(heterogeneous): no item.");
    }
}

std::vector<double> HeterogeneousConditionsESProductWrapper::select_pointer_d(const HGCalParameters* cpuObject, 
									const unsigned int& item) const {
  switch(item) 
    {
    case 0:
      return cpuObject->cellFineX_;
    case 1:
      return cpuObject->cellFineY_;
    case 2:
      return cpuObject->cellCoarseX_;
    case 3:
      return cpuObject->cellCoarseY_;
    default:
      throw std::runtime_error("HeterogeneousConditionsESProductWrapper::select_pointer_d(non-heterogeneous): no item.");
    }
}

int32_t*& HeterogeneousConditionsESProductWrapper::select_pointer_i(cp::HeterogeneousHGCalParameters* cpuObject, 
								    const unsigned int& item) const {
  switch(item) 
    {
    case 4:
      return cpuObject->waferTypeL_;
    default:
      throw std::runtime_error("HeterogeneousConditionsESProductWrapper::select_pointer_i(heterogeneous): no item.");
    }
}

std::vector<int32_t> HeterogeneousConditionsESProductWrapper::select_pointer_i(const HGCalParameters* cpuObject, 
									  const unsigned int& item) const {
  switch(item) 
    {
    case 4:
      return cpuObject->waferTypeL_;
    default:
      throw std::runtime_error("HeterogeneousConditionsESProductWrapper::select_pointer_i(non-heterogeneous): no item.");
    }
}

hgcal_conditions::HeterogeneousHEFConditionsESProduct const *HeterogeneousConditionsESProductWrapper::getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const {
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
	    gpuErrchk(cudaMallocHost(&data.host, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct)));
	    // Allocate the payload array(s) on device memory.
	    gpuErrchk(cudaMalloc(&(data.host->params.cellFineX_), chunk_));

	    // Allocate the payload object on the device memory.
	    gpuErrchk(cudaMalloc(&data.device, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct)));
	    // Transfer the payload, first the array(s) ...
	    gpuErrchk(cudaMemcpyAsync(data.host->params.cellFineX_, this->params_.cellFineX_, chunk_, cudaMemcpyHostToDevice, stream));
	    
	    for(unsigned int j=0; j<this->sizes_.size()-1; ++j)
	      {
		if( cp::types[j] == cp::HeterogeneousHGCalParametersType::Double and 
		    cp::types[j+1] == cp::HeterogeneousHGCalParametersType::Double )
		  select_pointer_d(&(data.host->params), j+1) = select_pointer_d(&(data.host->params), j) + this->sizes_[j];
		else if( cp::types[j] == cp::HeterogeneousHGCalParametersType::Double and 
			 cp::types[j+1] == cp::HeterogeneousHGCalParametersType::Int32_t )
		  select_pointer_i(&(data.host->params), j+1) = reinterpret_cast<int32_t*>( select_pointer_d(&(data.host->params), j) + this->sizes_[j] );
		else
		  throw std::runtime_error("HeterogeneousConditionsESProductWrapper::getHeterogeneousConditionsESProductAsync(): compare this functions' logic with hgcal_conditions::parameters::types.");
	      }

	    // ... and then the payload object
	    gpuErrchk(cudaMemcpyAsync(data.device, data.host, sizeof(hgcal_conditions::HeterogeneousHEFConditionsESProduct), cudaMemcpyHostToDevice, stream));
	  }); //gpuData_.dataForCurrentDeviceAsync

  // Returns the payload object on the memory of the current device
  return data.device;
}

// Destructor frees all member pointers
HeterogeneousConditionsESProductWrapper::GPUData::~GPUData() {
  if(host != nullptr) 
    {
      gpuErrchk(cudaFree(host->params.cellFineX_));
      gpuErrchk(cudaFreeHost(host));
    }
  gpuErrchk(cudaFree(device));
}

//template double*& HeterogeneousConditionsESProductWrapper::select_pointer_d<cp::HeterogeneousHGCalParameters*>(cp::HeterogeneousHGCalParameters*, const unsigned int&) const;
