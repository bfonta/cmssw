#ifndef DataFormats_HGCalReco_interface_BarrelSoARecHits_h
#define DataFormats_HGCalReco_interface_BarrelSoARecHits_h

#include "DataFormats/HGCalReco/interface/HGCalSoARecHits.h"

// The barrel (ECAL and HCAL) rechits share the layout of the HGCAL ones, with (dim1, dim2, dim3) = (eta, phi, r), r
// being the distance of the cell from the origin
using BarrelSoARecHits = HGCalSoARecHits;

#endif  // DataFormats_HGCalReco_interface_BarrelSoARecHits_h
