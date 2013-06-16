#ifndef _UTILITY_HPP
#define _UTILITY_HPP

#include <cmath>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>


inline double norm_pdf(double x) {
  using namespace std;
  return exp(-pow(x/2, 2)) / sqrt(2*boost::math::constants::pi<double>());
}

inline double norm_cdf(double x) {
  using namespace std;
  return 0.5 * boost::math::erfc(-x/(sqrt(2)));
}

inline double norm_cdfc(double x) {
  using namespace std;
  return 0.5 * boost::math::erfc(x/(sqrt(2)));
}

inline double norm_quantile(double p) {
  return -sqrt(2) * boost::math::erfc_inv(2*p);
}

class normal_distribution {

  boost::random::normal_distribution<double> dist;

public:

  normal_distribution(double mu, double sigma) : dist(mu, sigma) {}

  double mean() const { return dist.mean(); }

  double sigma() const { return dist.sigma(); }

  double pdf(double x) const {
    return norm_pdf((x-mean())/sigma());
  }

  double cdf(double x) const {
    return norm_cdf((x-mean())/sigma());
  }

  double cdfc(double x) const {
    return norm_cdfc((x-mean())/sigma());
  }

  template <class Engine> double operator()(Engine& rng) { return dist(rng); }

};


class trunc_normal_distribution {

  double _mu, _sigma;
  double alpha, beta;

  double cdf_alpha, cdf_beta;
  double delta, log_delta;

public:

  trunc_normal_distribution(double mu, double sigma, double a, double b)
    : _mu(mu), _sigma(sigma), alpha((a-mu)/sigma), beta((b-mu)/sigma),
      cdf_alpha(norm_cdf(alpha)), cdf_beta(norm_cdf(beta)),
      delta(cdf_beta - cdf_alpha)
  {}

  double mu() const { return _mu; }

  double sigma() const { return _sigma; }

  double pdf(double x) const {
    return norm_pdf((x-mu())/sigma()) / delta;
  }

  double cdf(double x) const {
    return (norm_cdf((x-mu())/sigma()) - cdf_alpha) / delta;
  }

  template <class Engine> double operator()(Engine& rng) const {
    double cdf_x = boost::random::uniform_real_distribution<double>(cdf_alpha, cdf_beta)(rng);
    return norm_quantile(cdf_x) * sigma() + mu();
  }

};

#endif
