#ifndef _TILE_CODER_HPP
#define _TILE_CODER_HPP

#include <cmath>
#include <vector>
#include <Eigen/Core>

using Eigen::VectorXd;
using Eigen::VectorXi;

class tile_coder_base {
public:
  virtual VectorXi indices (const VectorXd& coord) const = 0;
  virtual const VectorXd& get_feature_weights() const = 0;
  virtual int get_active_features() const = 0;
  virtual int get_num_features() const = 0;
};


class tile_coder : public tile_coder_base {

  struct tiling {

    VectorXd cell_size;
    VectorXi num_cells;
    VectorXd offset;
    int num_tiles;

    tiling (const VectorXd& cell_size, const VectorXi& num_cells, const VectorXd& offset)
      : cell_size(cell_size), num_cells(num_cells), offset(offset) {
      num_tiles = num_cells.prod();
    }

    int index(const VectorXd& coord) const {
      int index = 0;
      int base = 1;
      for (int i = 0; i < coord.size(); ++i) {
        int c = int(std::floor((coord(i) - offset(i)) / cell_size(i))) % num_cells(i);
        if (c < 0) c += num_cells(i);
        index += c * base;
        base *= num_cells(i);
      }
      return index;
    }

  };

  std::vector<tiling> tilings;
  VectorXd feature_weights;
  int num_features;

public:

  tile_coder (const VectorXd& cell_size, const VectorXi& num_cells, const VectorXi& num_offsets,
              const std::vector<int>& subspace_dims, double weight_exponent);

  VectorXi indices (const VectorXd& coord) const;

  const VectorXd& get_feature_weights () const { return feature_weights; }
  int get_active_features () const { return feature_weights.size(); }
  int get_num_features () const { return num_features; }

};


struct hashing_tile_coder : public tile_coder_base {

  tile_coder tc;
  int num_features;
  int hash_constant;

  hashing_tile_coder (const tile_coder& tc, int n)
    : tc(tc),
      num_features(std::min(n, tc.get_num_features())),
      hash_constant(n < tc.get_num_features() ? 2654435761 : 1)
  { }

  VectorXi indices (const VectorXd& coord) const;
  const VectorXd& get_feature_weights () const { return tc.get_feature_weights(); }
  int get_active_features () const { return tc.get_active_features(); }
  int get_num_features () const { return num_features; }

};

#endif
