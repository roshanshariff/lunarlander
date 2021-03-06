#ifndef _LUNAR_LANDER_AGENT_HPP
#define _LUNAR_LANDER_AGENT_HPP

#include <vector>
#include <random>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Core>

#include "simulator.hpp"
#include "policy_gradient_agent.hpp"
#include "tile_coder.hpp"

using Eigen::VectorXd;

class lunar_lander_agent {

  VectorXd max_state, min_state, max_clip_state, min_clip_state;

  hashing_tile_coder tc;
  td_critic critic;
  policy_gradient_actor thrust_actor, rcs_actor;

  tile_coder make_tile_coder (double tile_weight_exponent, const std::vector<int>& subspaces);
  static policy_gradient_actor make_thrust_actor (const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal);
  static policy_gradient_actor make_rcs_actor (const tile_coder_base& tc, double lambda, double alpha, bool trunc_normal);

  lunar_lander_simulator::action compute_action(std::mt19937& rng, const VectorXi& features) {
    return lunar_lander_simulator::action(thrust_actor.act(rng, features), rcs_actor.act(rng, features));
  }

  void clip_state(VectorXd& state) {
    for (unsigned int i = 0; i < state.size(); ++i) {
      state(i) = std::max(min_state(i), std::min(state(i), max_state(i)));
    }
  }

public:

  lunar_lander_agent(double lambda, double alpha_v, double alpha_u, double initial_value,
                     int num_features,
                     double tile_weight_exponent,
                     bool trunc_normal,
                     const std::vector<int>& subspaces)
    : max_state(6), min_state(6), max_clip_state(6), min_clip_state(6),
      tc (make_tile_coder (tile_weight_exponent, subspaces), num_features),
      critic (tc, lambda, alpha_v, initial_value),
      thrust_actor (make_thrust_actor (tc, lambda, alpha_u, trunc_normal)),
      rcs_actor (make_rcs_actor (tc, lambda, alpha_u, trunc_normal))
  { }

  lunar_lander_simulator::action initialize(std::mt19937& rng, VectorXd state);

  lunar_lander_simulator::action update(std::mt19937& rng, VectorXd state,
                                        double reward, bool terminal=false, bool learn=true);

  const VectorXd& get_max_state() const { return max_state; }
  const VectorXd& get_min_state() const { return min_state; }

};

#endif
