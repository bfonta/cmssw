#ifndef CUDADAtaFormats_HGCal_HGCRecHitHost_H
#define CUDADAtaFormats_HGCal_HGCRecHitHost_H

#include <cassert>
#include <numeric>
#include <memory>

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"

std::unique_ptr<HGCRecHitSoA> layoutHGCRecHitHost(uint32_t nhits, const cudaStream_t& stream);

#endif  //CUDADAtaFormats_HGCal_HGCRecHitHost_H
