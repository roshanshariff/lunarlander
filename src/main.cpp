#include <boost/nondet_random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <iostream>
#include <limits>

#include "utility.hpp"

int main () {
  boost::random::random_device rand_dev;
  boost::random::mt19937 rng(rand_dev());

  //double inf = std::numeric_limits<double>::infinity();
  trunc_normal_distribution dist(2, 5, -2, 2);

  for (double x = -10; x <= 10; x += 0.001) {
    std::cout << x << " " << dist.pdf(x) << "\n";
  }

  return 0;
}
