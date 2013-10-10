#include <numeric>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

#include "utility.hpp"
#include "simulator.hpp"
#include "lunar_lander_agent.hpp"
#include "framework.hpp"

int main (int argc, char* argv[]) {

  unsigned int seed = std::random_device()();

  std::mt19937 agent_rng(seed);
  std::mt19937 init_rng(0);

  double dt = 0.05;
  int agent_time_steps = 10;

  int num_episodes = 20000;
  double lambda = 0.75;
  double alpha_v = 0.1;
  double alpha_u = 0.1;
  double initial_value = 1;
  int num_features = 1<<20;
  double tile_weight_exponent = 0.5; // 1 for no weighting
  bool trunc_normal = false;

  std::vector<int> subspaces { 0, 1, 2, 6 };

  bool visualize = false;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--dt") dt = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--agent_steps") agent_time_steps = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--episodes") num_episodes = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--lambda") lambda = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--alpha-v") alpha_v = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--alpha-u") alpha_u = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--initial-value") initial_value = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--trunc-normal") trunc_normal = true;
    else if (std::string(argv[i]) == "--no-trunc-normal") trunc_normal = false;
    else if (std::string(argv[i]) == "--tile-weight-exponent") tile_weight_exponent = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--visualize") visualize = true;
    else if (std::string(argv[i]) == "--subspaces") {
      std::istringstream arg (argv[++i]);
      subspaces.assign (std::istream_iterator<int>(arg), std::istream_iterator<int>());
    }
    else {
      std::fprintf (stderr, "Unknown parameter: %s\n",argv[i]);
      std::exit(1);
    }
  }

  framework f(lunar_lander_simulator(),
              lunar_lander_agent(lambda, alpha_v, alpha_u, initial_value, num_features,
                                 tile_weight_exponent, trunc_normal, subspaces),
              dt,
              agent_time_steps);

  for (int i = 0; i < num_episodes; i++) {
    if (visualize) f.set_visualiser (i % 1000 == 0 ? stdout : 0);
    std::vector<double> rewards = f.run_episode(init_rng, agent_rng);
    if (!visualize) std::fprintf(stdout, "%g\n", f.get_return());
    // std::printf("%g\n", f.time_elapsed);
  }

  return 0;
}
