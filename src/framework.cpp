#include <boost/math/constants/constants.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <cstdlib>
#include <cstdio>

#include "framework.hpp"

double framework::reward() {

  double reward = 0;

  double x_pos = std::abs (simulator.get_lander().get_pos().x());
  if (x_pos > 100) {
    simulator.set_crashed();
    x_pos = 100;
  }

  if (simulator.get_crashed() || simulator.get_landed()) {
    reward -= x_pos / lunar_lander_simulator::LANDER_WIDTH();
    if (simulator.get_landed()) reward += 1;
  }
  else if (time_elapsed > 600) {
    std::fprintf(stderr, "Time limit exceeded.\n");
    reward -= 10;
    simulator.set_crashed();
  }

  if (!simulator.get_landed()) {
    reward -= std::log10 (1 + simulator.get_lander().get_breakage());
    simulator.get_lander().reset_breakage();
  }

  reward -= 0.01 * simulator.get_main_throttle();

  return reward;
}


void framework::initialize_simulator(boost::random::mt19937& rng) {

  typedef boost::random::uniform_real_distribution<double> uniform;
  typedef boost::random::normal_distribution<double> normal;
  const double PI = boost::math::constants::pi<double>();

  double rotation = normal(0, PI/8)(rng);
  simulator.get_lander().set_rot(rotation);

  double xpos = uniform(agent.get_min_state()(0), agent.get_max_state()(0))(rng);
  double ypos = uniform(simulator.get_lander().get_min_y(), agent.get_max_state()(1))(rng);

  double xvel = 0;
  double yvel = 0;
  double rot_vel = 0;

  simulator.initialize(xpos, ypos, xvel, yvel, rotation, rot_vel);
}


VectorXd framework::simulator_state() const {

  VectorXd state(6);
  state.segment<2>(0) = simulator.get_lander().get_pos();
  state.segment<2>(2) = simulator.get_lander().get_vel();
  state(4) = simulator.get_lander().get_rot();
  state(5) = simulator.get_lander().get_rot_vel();

  if (state(0) < 0) {
    state(0) = -state(0);
    state(2) = -state(2);
    state(4) = -state(4);
    state(5) = -state(5);
  }

  return state;
}


void framework::take_action(lunar_lander_simulator::action a) {
  if (simulator.get_lander().get_pos().x() < 0) a.rcs = -a.rcs;
  simulator.set_action(a);
}


std::vector<double> framework::run_episode(boost::random::mt19937& rng) {

  initialize_simulator(rng);
  take_action(agent.initialize(rng, simulator_state()));

  time_elapsed = 0;
  std::vector<double> rewards;

  while (true) {
    for (int i = 0; i < agent_time_steps; ++i) {
      if (simulator.get_landed() || simulator.get_crashed()) break;
      simulator.update(dt);
      time_elapsed += dt;
    }

    double Reward = reward();
    rewards.push_back(Reward);

    bool terminal = simulator.get_crashed() || simulator.get_landed();
    take_action(agent.update(rng, simulator_state(), Reward, terminal));

    if (terminal) break;
  }

  return rewards;
}
