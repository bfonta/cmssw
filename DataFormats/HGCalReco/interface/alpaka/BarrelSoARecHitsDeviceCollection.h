#ifndef DataFormats_HGCalReco_interface_alpaka_BarrelSoARecHitsDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_BarrelSoARecHitsDeviceCollection_h

#include "DataFormats/HGCalReco/interface/BarrelSoARecHits.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The same type as HGCalSoARecHitsDeviceCollection, so it needs no dictionary of its own
  using BarrelSoARecHitsDeviceCollection = HGCalSoARecHitsDeviceCollection;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalReco_interface_alpaka_BarrelSoARecHitsDeviceCollection_h
