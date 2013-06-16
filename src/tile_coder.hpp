#ifndef _TILE_CODER_HPP
#define _TILE_CODER_HPP

#include <cmath>
#include <vector>
#include <Eigen/Core>

using Eigen::VectorXd;
using Eigen::VectorXi;

class tile_coder {

  class tiling {
    VectorXd cell_size;
    VectorXi num_cells;
    VectorXd offset;
    int num_tiles;

  public:

    tiling (const VectorXd& cell_size, const VectorXi& num_cells, const VectorXd& offset)
      : cell_size(cell_size), num_cells(num_cells), offset(offset) {
      num_tiles = num_cells.prod();
    }

    int index(const VectorXd& coord) {
      int index = 0;
      int base = 1;
      for (int i = 0; i < coord.size(); ++i) {
        int c = int(std::floor((coord(i) - offset(i)) / cell_size(i))) % num_cells(i);
        index += c * base;
        base *= num_cells(i);
      }
      return index;
    }

  };



};

#endif
