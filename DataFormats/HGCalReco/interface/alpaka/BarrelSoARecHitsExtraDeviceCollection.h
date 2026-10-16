#ifndef DataFormats_HGCalReco_interface_alpaka_BarrelSoARecHitsExtraDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_BarrelSoARecHitsExtraDeviceCollection_h

#include "DataFormats/HGCalReco/interface/BarrelSoARecHitsExtra.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The same type as HGCalSoARecHitsExtraDeviceCollection, so it needs no dictionary of its own
  using BarrelSoARecHitsExtraDeviceCollection = HGCalSoARecHitsExtraDeviceCollection;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalReco_interface_alpaka_BarrelSoARecHitsExtraDeviceCollection_h
