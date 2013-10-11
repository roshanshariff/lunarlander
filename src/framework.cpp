#include <boost/math/constants/constants.hpp>
#include <random>
#include <cstdlib>
#include <cstdio>

#include "framework.hpp"

double framework::reward() {

  double reward = -0.05 * simulator.get_main_throttle();

  double x_pos = std::abs (simulator.get_lander().get_pos().x());
  if (x_pos > 100) {
    simulator.set_crashed();
    x_pos = 100;
  }

  const bool episode_terminated = simulator.get_crashed() || simulator.get_landed();
  const bool time_exceeded = time_elapsed > 180;

  if (episode_terminated || time_exceeded) {

    reward -= x_pos / lunar_lander_simulator::LANDER_WIDTH();

    if (simulator.get_landed()) reward += 1;

    if (!episode_terminated && time_exceeded) {
      std::fprintf(stderr, "Time limit exceeded.\n");
      simulator.set_crashed();
      reward -= 1;
    }
  }

  if (!simulator.get_landed()) {
    reward -= std::log10 (1 + simulator.get_lander().get_breakage());
    simulator.get_lander().reset_breakage();
  }

  return reward;
}


void framework::initialize_simulator(std::mt19937& rng) {

  typedef std::uniform_real_distribution<double> uniform;
  typedef std::normal_distribution<double> normal;
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


void framework::run_episode(std::mt19937& init_rng, std::mt19937& agent_rng) {

  initialize_simulator(init_rng);
  take_action(agent.initialize(agent_rng, simulator_state()));

  _return = 0;
  rewards.clear();
  time_elapsed = 0;

  bool terminal = false;

  while (!terminal) {

    for (int i = 0; i < agent_time_steps && !terminal; ++i) {
      send_visualiser_data();
      simulator.update(dt);
      time_elapsed += dt;
      terminal = simulator.get_crashed() || simulator.get_landed();
    }

    double _reward = reward();
    _return += _reward;
    rewards.push_back(_reward);

    take_action(agent.update(agent_rng, simulator_state(), _reward, terminal));
  }

  send_visualiser_data();
  if (visualiser) std::fflush (visualiser);
}


void framework::send_visualiser_data () const {

  if (!visualiser) return;

  const Vector2d& pos = simulator.get_lander().get_pos();
  const Vector2d& vel = simulator.get_lander().get_vel();

  std::fprintf (visualiser,
                "{ \"Return\": %g, \"x\": %g, \"y\": %g, \"vx\": %g, \"vy\": %g, \"rot\": %g, \"vrot\": %g, "
                "\"thrust\": %g, \"rcs\": %g, \"breakage\": %g, \"crashed\": %s, \"landed\": %s }\n",
                get_return(), pos.x(), pos.y(), vel.x(), vel.y(),
                simulator.get_lander().get_rot(), simulator.get_lander().get_rot_vel(),
                simulator.get_action().thrust, simulator.get_action().rcs,
                simulator.get_lander().get_breakage(),
                simulator.get_crashed() ? "true" : "false",
                simulator.get_landed() ? "true" : "false");
}
