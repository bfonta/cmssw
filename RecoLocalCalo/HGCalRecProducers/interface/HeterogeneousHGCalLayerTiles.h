#ifndef HeterogeneousHGCalLayerTiles_h
#define HeterogeneousHGCalLayerTiles_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>
//GPU Add
#include <cuda_runtime.h>
#include <cuda.h>

#include "GPUVecArray.h"
#include "HGCalTilesConstants.h"


class HeterogeneousHGCalLayerTiles {

  using LTC = HGCalTilesConstants;
  
  public:

    // constructor
    HeterogeneousHGCalLayerTiles(){};

    __device__
    void fill(float x, float y, int i)
    {
      layerTiles_[getGlobalBin(x,y)].push_back(i);
    }

    __host__ __device__
    int getXBin(float x) const {
      int xBin = (x-LTC::minX)*LTC::rX;
      xBin = (xBin<LTC::nColumns ? xBin:LTC::nColumns-1);
      xBin = (xBin>0 ? xBin:0);
      return xBin;
    }

    __host__ __device__
    int getYBin(float y) const {
      int yBin = (y-LTC::minY)*LTC::rY;
      yBin = (yBin<LTC::nRows ? yBin:LTC::nRows-1);
      yBin = (yBin>0 ? yBin:0);;
      return yBin;
    }

    __host__ __device__
    int getGlobalBin(float x, float y) const{
      return getXBin(x) + getYBin(y)*LTC::nColumns;
    }

    __host__ __device__
    int getGlobalBinByBin(int xBin, int yBin) const {
      return xBin + yBin*LTC::nColumns;
    }

    __host__ __device__
    int4 searchBox(float xMin, float xMax, float yMin, float yMax){
      return int4{ getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)};
    }

    __host__ __device__
    void clear() {
      for(auto& t: layerTiles_) t.reset();
    }

    __host__ __device__
    cms::cuda::VecArray<int, LTC::maxTileDepth>& operator[](int globalBinId) {
      return layerTiles_[globalBinId];
    }



  private:
    cms::cuda::VecArray<cms::cuda::VecArray<int, LTC::maxTileDepth>, LTC::nColumns * LTC::nRows > layerTiles_;
};
#endif //HeterogeneousHGCalLayerTiles_h
