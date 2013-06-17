#ifndef _FRAMEWORK_HPP
#define _FRAMEWORK_HPP

#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Core>
#include <vector>
#include <cstdio>

#include "simulator.hpp"
#include "lunar_lander_agent.hpp"

using Eigen::VectorXd;

struct framework {

  lunar_lander_simulator simulator;
  lunar_lander_agent agent;
  double dt, time_elapsed, _return;
  int agent_time_steps;

  FILE* visualiser;

public:

  framework(const lunar_lander_simulator& simulator, const lunar_lander_agent& agent, double dt, int agent_time_steps)
    : simulator(simulator), agent(agent), dt(dt), agent_time_steps(agent_time_steps), visualiser(0) { }

  double reward();

  void initialize_simulator(boost::random::mt19937& rng);

  VectorXd simulator_state() const;

  void take_action(lunar_lander_simulator::action a);

  std::vector<double> run_episode(boost::random::mt19937& init_rng,
                                  boost::random::mt19937& agent_rng);

  double get_return () const { return _return; }

  void set_visualiser (FILE* output) { visualiser = output; }
  void send_visualiser_data () const;

};

#endif
