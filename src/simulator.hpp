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

  void set_breakage (double new_breakage) {
    breakage = std::max (breakage, new_breakage);
  }

public:

  void apply_impulse (Vector2d position, Vector2d impulse) {
    vel += impulse / mass;
    rot_vel += (position.x()*impulse.y() - position.y()*impulse.x()) / mom_inertia;
  }

  void update (const double dt, const Vector2d& force, const double torque);

  void clear_breakage () {
    breakage = 0.0;
  }

};




#endif
