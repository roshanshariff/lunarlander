#include <vector>
#include <iostream>

#include <Eigen/Core>

#include "tile_coder.hpp"


namespace {


  void select_combination (std::vector<std::vector<int> >& result, std::vector<int>& partial,
                           int from, int to, unsigned int index) {

    if (index >= partial.size()) {
      result.push_back (partial);
      return;
    }

    for (int selection = from; selection < to; ++selection) {
      partial[index] = selection;
      select_combination (result, partial, selection+1, to, index+1);
    }
  }


  void subspace_offsets (std::vector<VectorXi>& result, VectorXi& partial, const VectorXi& num_offsets,
                         unsigned int index) {

    if (index >= partial.size()) {
      result.push_back (partial);
      return;
    }

    for (partial(index) = 0; partial(index) < num_offsets(index); ++partial(index)) {
      subspace_offsets (result, partial, num_offsets, index+1);
    }
  }


}


tile_coder::tile_coder (const VectorXd& cell_size, const VectorXi& num_cells, const VectorXi& num_offsets,
                        const std::vector<int>& subspace_dims, double weight_exponent)
  : num_features(0) {

  const int space_dim = cell_size.size();

  std::vector<std::vector<int> > subspace_selections;

  for (unsigned int i = 0; i < subspace_dims.size(); ++i) {
    std::vector<int> partial (subspace_dims[i]);
    select_combination (subspace_selections, partial, 0, space_dim, 0);
  }

  std::vector<double> vec_feature_weights;

  for (unsigned int i = 0; i < subspace_selections.size(); ++i) {

    const std::vector<int>& subspace = subspace_selections[i];

    VectorXi subspace_num_cells = VectorXi::Ones (space_dim);
    VectorXi subspace_num_offsets = VectorXi::Ones (space_dim);

    {
      for (unsigned int j = 0; j < subspace.size(); ++j) {

        const int coord_ix = subspace[j];

        subspace_num_cells[coord_ix] = num_cells[coord_ix];
        subspace_num_offsets[coord_ix] = num_offsets[coord_ix];
      }

      int num_tilings = subspace_num_offsets.prod();
      int num_tiles = subspace_num_cells.prod();
      num_features += num_tilings * num_tiles;

      double feature_weight = std::pow (num_tilings, -weight_exponent);
      for (int i = 0; i < num_tilings; ++i) vec_feature_weights.push_back (feature_weight);
    }

    std::vector<VectorXi> offset_list;
    {
      VectorXi partial = VectorXi::Zero (space_dim);
      subspace_offsets (offset_list, partial, subspace_num_offsets, 0);
    }

    for (unsigned int i = 0; i < offset_list.size(); ++i) {
      VectorXd offset = cell_size.array() * offset_list[i].cast<double>().array() / subspace_num_offsets.cast<double>().array();
      tilings.push_back (tiling (cell_size, subspace_num_cells, offset));
    }
  }

  feature_weights = VectorXd::Map (&vec_feature_weights[0], vec_feature_weights.size());
  feature_weights.normalize();

}


VectorXi tile_coder::indices (const VectorXd& coord) const {

  VectorXi result (tilings.size());

  int start_index = 0;
  for (unsigned int i = 0; i < tilings.size(); ++i) {
    result(i) = start_index + tilings[i].index (coord);
    start_index += tilings[i].num_tiles;
  }

  return result;

}


VectorXi hashing_tile_coder::indices (const VectorXd& coord) const {
  VectorXi result = tc.indices (coord);
  for (int i = 0; i < result.size(); ++i) {
    result(i) *= hash_constant;
    result(i) %= num_features;
    if (result(i) < 0) result(i) += num_features;
  }
  return result;
}


// int main () {

//   std::vector<int> subspace_dims;
//   subspace_dims.push_back(1);
//   subspace_dims.push_back(2);


//   hashing_tile_coder tiler (tile_coder (Eigen::Vector2d(0.5, 0.5), Eigen::Vector2i(2,2), Eigen::Vector2i(2,2), subspace_dims, 1.0), 4);


//   std::cout << "Num features:        " << tiler.get_num_features() << std::endl;
//   std::cout << "Num active features: " << tiler.get_active_features() << std::endl;
//   std::cout << "Weights:             " << tiler.get_feature_weights().transpose() << std::endl;

//   double x, y;
//   while (std::cin >> x >> y) {
//     VectorXi result = tiler.indices (Eigen::Vector2d (x, y));
//     for (int i = 0; i < result.size(); ++i) {
//       std::cout << result(i) << ' ';
//     }
//     std::cout << std::endl;
//   }

// }
