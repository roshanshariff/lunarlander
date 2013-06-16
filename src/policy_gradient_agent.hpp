#ifndef _POLICY_GRADIENT_AGENT_HPP
#define _POLICY_GRADIENT_AGENT_HPP

#include <cmath>
#include <Eigen/Core>
#include <deque>
#include <utility>

#include "tile_coder.hpp"

using Eigen::VectorXd;

class linear_function_approx {

  VectorXd feature_weights;
  VectorXd weights;
  double falloff;
  unsigned int trace_len;
  std::deque<std::pair<VectorXi, double> > trace;

public:

  linear_function_approx(const tile_coder_base& tc, double falloff, double initial_value=0, double threshold=0.05)
    : feature_weights(tc.get_feature_weights()),
      weights(VectorXd::Constant(tc.get_num_features(), initial_value/feature_weights.sum())),
      falloff(falloff),
      trace_len(falloff > 0 ? (unsigned int)(std::ceil(std::log(threshold) / std::log(falloff))) : 1)
  {}

  void initialize() { trace.clear(); }

  void add_features(const VectorXi& features, double scaling=1.0);

  double value(const VectorXi& features)  const;

  void update(double delta);
};


class td_critic {

  double alpha;
  double gamma;
  linear_function_approx value;
  VectorXi features;

public:

  td_critic(const tile_coder_base& tc, double lambda, double alpha, double initial_value=0, double gamma=1)
    : alpha(alpha), gamma(gamma), value(tc, gamma*lambda, initial_value) {}

  void initialize(const VectorXi& new_features) {
    features = new_features;
    value.initialize();
  }

  double evaluate(const VectorXi& new_features, double reward, bool terminal=false);

};

class policy_gradient_actor {

public:

  void initialize();


};

#endif
