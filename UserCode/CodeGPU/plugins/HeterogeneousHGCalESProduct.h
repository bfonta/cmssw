#ifndef HeterogeneousHGCalESProduct_h
#define HeterogeneousHGCalESProduct_h

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "UserCode/CodeGPU/plugins/KernelManagerHGCalRecHit.h"

// Declare the wrapper ESProduct. The corresponding ESProducer should
// produce objects of this type.
class HeterogeneousConditionsESProductWrapper {
 public:
  // Constructor takes the standard CPU ESProduct, and transforms the
  // necessary data to array(s) in pinned host memory
  HeterogeneousConditionsESProductWrapper(unsigned int const&, const HGCalDDDConstants*);
  
  // Deallocates all pinned host memory
  ~HeterogeneousConditionsESProductWrapper();
  
  // Function to return the actual payload on the memory of the current device
  HeterogeneousHEFConditionsESProduct const *getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const;
  
 private:
  // Holds the data in pinned CPU memory
  HeterogeneousHGCalDDDConstants ddd_;
  unsigned int chunk1_;

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    // internal pointers are on device, struct itself is on CPU
    HeterogeneousHEFConditionsESProduct *host = nullptr;
    // internal pounters and struct are on device
    HeterogeneousHEFConditionsESProduct *device = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
};

#endif //HeterogeneousHGCalESProduct_h
