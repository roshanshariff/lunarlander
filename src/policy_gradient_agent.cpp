#include "policy_gradient_agent.hpp"


void linear_function_approx::add_features(const VectorXi& features, double scaling) {
  if (trace.size() >= trace_len) trace.pop_back();
  trace.push_front(std::make_pair(features, scaling));
}


double linear_function_approx::value(const VectorXi& features) const {
  double value = 0;
  for (int i = 0; i < features.size(); ++i) {
    value += weights(features(i)) * feature_weights(i);
  }
  return value;
}


void linear_function_approx::update(double delta) {
  for (unsigned int i = 0; i < trace.size(); ++i) {
    const VectorXi& features = trace[i].first;
    const double amount = delta * trace[i].second;
    for (int j = 0; j < features.size(); ++j) {
      weights(features(j)) += amount * feature_weights(j);
    }
    delta *= falloff;
  }
}


double td_critic::evaluate(const VectorXi& new_features, double reward, bool terminal) {
  double old_value = value.value (features);
  double new_value = terminal ? 0.0 : value.value (new_features);
  double td_error = reward + gamma*new_value - old_value;

  value.add_features(features);
  value.update(alpha * td_error);

  features = new_features;

  return td_error;
}
