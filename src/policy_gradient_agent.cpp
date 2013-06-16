#include <boost/math/special_functions/fpclassify.hpp>
#include <limits>

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


trunc_normal_distribution policy_gradient_actor::action_dist() const {

  double mu_value = mu.value(features);
  double sigma_value = sigma.value(features);
  sigma_value = min_sigma + sigma_range * (1 + std::tanh(sigma_value/2)) / 2;

  const double max_mu = max_action + 3.0 * sigma_value;
  const double min_mu = min_action - 3.0 * sigma_value;
  mu_value = std::min (std::max (min_mu, mu_value), max_mu);

  const double inf = std::numeric_limits<double>::infinity();
  if (!use_trunc_normal) return trunc_normal_distribution(mu_value, sigma_value, -inf, inf);
  else return trunc_normal_distribution(mu_value, sigma_value, min_action, max_action);
}


void policy_gradient_actor::learn(const double td_error) {

  const trunc_normal_distribution dist = action_dist();
  const double std_action = (action - dist.mu()) / dist.sigma();

  const double min_action_pdf = norm_pdf(dist.alpha()) / dist.norm_constant();
  const double max_action_pdf = norm_pdf(dist.beta()) / dist.norm_constant();
  const double trunc_grad_mu = use_trunc_normal ? max_action_pdf - min_action_pdf : 0;
  double trunc_grad_sigma = use_trunc_normal ? dist.beta()*max_action_pdf - dist.alpha()*min_action_pdf : 0;

  if (!boost::math::isfinite(trunc_grad_sigma)) trunc_grad_sigma = 0;

  const double variance = std::pow(dist.sigma(),2) * (1 - trunc_grad_sigma - std::pow(trunc_grad_mu,2));
  const double scaled_alpha = alpha * variance;

  const double mu_grad = (std_action + trunc_grad_mu) / dist.sigma();
  mu.add_features(features, mu_grad);
  mu.update(scaled_alpha * td_error);

  const double sigma_grad = (std::pow(std_action,2) - 1 + trunc_grad_sigma) /
    dist.sigma() * (dist.sigma() - min_sigma) *
    (1 - (dist.sigma() - min_sigma) / sigma_range);
  sigma.add_features(features, sigma_grad);
  sigma.update(scaled_alpha * td_error);
}
