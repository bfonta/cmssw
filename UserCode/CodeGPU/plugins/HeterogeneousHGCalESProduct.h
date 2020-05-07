#ifndef HeterogeneousHGCalESProduct_h
#define HeterogeneousHGCalESProduct_h

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "CUDADataFormats/HGCal/interface/HGCConditions.h"

// Declare the wrapper ESProduct. The corresponding ESProducer should
// produce objects of this type.
class HeterogeneousConditionsESProductWrapper {
 public:
  // Constructor takes the standard CPU ESProduct, and transforms the
  // necessary data to array(s) in pinned host memory
  HeterogeneousConditionsESProductWrapper(int const&, HeterogeneousConditionsESProduct const&);
  
  // Deallocates all pinned host memory
  ~HeterogeneousConditionsESProductWrapper();
  
  // Function to return the actual payload on the memory of the current device
  HeterogeneousConditionsESProduct const *getHeterogeneousConditionsESProductAsync(cudaStream_t stream) const;
  
 private:
  // Holds the data in pinned CPU memory
  int *layer_;
  int *wafer_;
  size_t nelements_;
  size_t chunk_;

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    // internal pointers are on device, struct itself is on CPU
    HeterogeneousConditionsESProduct *host = nullptr;
    // internal pounters and struct are on device
    HeterogeneousConditionsESProduct *device = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
};

#endif //HeterogeneousHGCalESProduct_h
