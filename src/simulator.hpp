#ifndef _SIMULATOR_HPP
#define _SIMULATOR_HPP

#include <vector>

#include <Eigen/Core>

using Eigen::Vector2d;
using Eigen::Matrix2d;

class rigid_body {

public:

  struct collider {

    Vector2d pos;
    Matrix2d strength;

    Vector2d impulses;
    bool collided;
    bool contacted;

    collider (const Vector2d& pos, const Matrix2d& strength) : pos(pos), strength(strength) {
      reset_collision();
    }

    void reset_collision () {
      impulses.setZero();
      collided = false;
      contacted = false;
    }

  };

private:

  Eigen::Vector2d pos, vel;
  double rot, rot_vel;

  double mass, mom_inertia;

  double restitution, mu_s, mu_k;

  std::vector<collider> colliders;
  double bounding_radius;

  double breakage;

  bool process_collisions (const double restitution,
                           const std::vector<Vector2d>& colliders_dpos_dtheta,
                           const Vector2d& new_pos,
                           const double new_rot,
                           const bool processing_contact);

  void accumulate_breakage (double new_breakage) {
    breakage = std::max (breakage, new_breakage);
  }

public:

  rigid_body(double mass, double mom_inertia, double mu_s, double mu_k, double restitution,
             const std::vector<collider>& colliders)
    : mass(mass), mom_inertia(mom_inertia), mu_s(mu_s), mu_k(mu_k), restitution(restitution), colliders(colliders) {}

  void apply_impulse (Vector2d position, Vector2d impulse) {
    vel += impulse / mass;
    rot_vel += (position.x()*impulse.y() - position.y()*impulse.x()) / mom_inertia;
  }

  void update (const double dt, const Vector2d& force, const double torque);

  void reset_breakage () {
    breakage = 0.0;
  }

  void reset_collisions () {
    for (int i = 0; i < colliders.size(); i++) colliders[i].reset_collision();
  }

  double get_min_y();

  const std::vector<collider>& get_colliders() { return colliders; }

  void set_pos(const Vector2d& new_pos) {
    pos = new_pos;
    pos.y() = std::max(get_min_y(), pos.y());
  }

  const Vector2d& get_pos() { return pos; }

  void set_vel(const Vector2d& new_vel) { vel = new_vel; }
  const Vector2d& get_vel() { return vel; }

  void set_rot(double new_rot) {
    rot = new_rot;
    pos.y() = std::max(get_min_y(), pos.y());
  }
  double get_rot() { return rot; }

  void set_rot_vel(double new_rot_vel) { rot_vel = new_rot_vel; }
  double get_rot_vel() { return rot_vel; }

  double get_mass() { return mass; }

  double get_mom_inertia() { return mom_inertia; }

  double get_breakage() { return breakage; }

};


class lunar_lander_simulator {

  rigid_body lander;
  double dt;

  bool crashed, landed;
  double thrust, rcs;

  // http://www.ibiblio.org/apollo/NARA-SW/R-567-sec6.pdf
  // Assuming descent stage 50% fuel, ascent stage 100% fuel

  static const double GRAVITY = 1.622; // m/s^2
  static const double LANDER_WIDTH = 9.07; // m

  static const double MASS = 11036.4; // kg
  static const double MOM_INERTIA = 28258.7; // kg m^2
  static const double RESTITUTION = 0.2;
  static const double MU_S = 1.0;
  static const double MU_K = 0.9;

  static const double MAX_RCS = 0.16056757359680382; // rad/s^2
  static const double MAX_THRUST = 4.081113406545613; // m/s^2

  static inline Vector2d IMAGE_TO_BODY_COORDS (double x, double y) {
    return Vector2d(x - 0.5, y - 0.36335788) * 9.07;
  }

  static rigid_body::collider MAKE_LEG_COLLIDER (double x, double y, double strut_dir);
  static rigid_body::collider MAKE_BODY_COLLIDER (double x, double y);
  static std::vector<rigid_body::collider> MAKE_COLLIDERS();

public:

  lunar_lander_simulator (double dt);

  void initialize(double pos_x=0, double pos_y=0, double vel_x=0, double vel_y=0, double rot=0, double rot_vel=0);

  void update();

  void set_action(double thrust, double rcs);
};

#endif
