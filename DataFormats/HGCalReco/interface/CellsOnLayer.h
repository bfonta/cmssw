#ifndef DataFormats_HGCalReco_CellsOnLayer_h
#define DataFormats_HGCalReco_CellsOnLayer_h

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

struct CellsOnLayer {
  std::vector<DetId> detid;
  std::vector<bool> isSi;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> eta;
  std::vector<float> phi;

  std::vector<float> weight;
  std::vector<float> rho;

  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<int> clusterIndex;
  std::vector<float> sigmaNoise;
  std::vector<std::vector<int>> followers;
  std::vector<bool> isSeed;

  void clear() {
    detid.clear();
    isSi.clear();
    x.clear();
    y.clear();
    eta.clear();
    phi.clear();
    weight.clear();
    rho.clear();
    delta.clear();
    nearestHigher.clear();
    clusterIndex.clear();
    sigmaNoise.clear();
    followers.clear();
    isSeed.clear();
  }

  void shrink_to_fit() {
    detid.shrink_to_fit();
    isSi.shrink_to_fit();
    x.shrink_to_fit();
    y.shrink_to_fit();
    eta.shrink_to_fit();
    phi.shrink_to_fit();
    weight.shrink_to_fit();
    rho.shrink_to_fit();
    delta.shrink_to_fit();
    nearestHigher.shrink_to_fit();
    clusterIndex.shrink_to_fit();
    sigmaNoise.shrink_to_fit();
    followers.shrink_to_fit();
    isSeed.shrink_to_fit();
  }
};

#endif //DataFormats_HGCalReco_CellsOnLayer_h
