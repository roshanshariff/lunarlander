#include <numeric>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <regex>
#include <iostream>

#include "utility.hpp"
#include "simulator.hpp"
#include "lunar_lander_agent.hpp"
#include "framework.hpp"

int main (int argc, char* argv[]) {

  std::random_device rdev;

  unsigned int agent_seed = rdev();
  unsigned int init_seed = rdev();


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
  int visualize_from = 0;
  int visualize_every = 1000;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--agent-seed") agent_seed = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--init-seed") init_seed = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--dt") dt = std::atof(argv[++i]);
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
    else if (std::string(argv[i]) == "--visualize-from") visualize_from = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--visualize-every") visualize_every = std::atoi(argv[++i]);
    else if (std::string(argv[i]) == "--subspaces") {
      subspaces.clear();
      std::istringstream arg(argv[++i]);
      int dim;
      char sep;
      while (arg >> dim) {
        subspaces.push_back(dim);
        if (arg >> sep && sep != ',') {
          fprintf(stderr, "Invalid subspaces argument\n");
          return 1;
        }
      }
    }
    else {
      std::fprintf (stderr, "Unknown parameter: %s\n",argv[i]);
      std::exit(1);
    }
  }

  std::mt19937 agent_rng(agent_seed);
  std::mt19937 init_rng(init_seed);
  if (!visualize) std::fprintf(stdout, "# agent-seed = %u\n# init-seed = %u\n", agent_seed, init_seed);

  framework f(lunar_lander_simulator(),
              lunar_lander_agent(lambda, alpha_v, alpha_u, initial_value, num_features,
                                 tile_weight_exponent, trunc_normal, subspaces),
              dt,
              agent_time_steps);

  for (int i = 0; i < num_episodes; i++) {

    if (visualize && i >= visualize_from) {
      f.set_visualiser ((i - visualize_from) % visualize_every == 0 ? stdout : nullptr);
    }

    f.run_episode(init_rng, agent_rng);

    if (!visualize) std::fprintf(stdout, "%g\n", f.get_return());
    // std::printf("%g\n", f.time_elapsed);
  }

  return 0;
}
