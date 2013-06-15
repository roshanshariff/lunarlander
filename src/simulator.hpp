#ifndef _SIMULATOR_HPP
#define _SIMULATOR_HPP

#include <vector>

#include <Eigen/Core>

using Eigen::Vector2d;
using Eigen::Matrix2d;

class rigid_body {

public:

  struct collider {

    Vector3d pos;
    Matrix2d strength;

    collider (const Vector2d& pos, const Matrix2d& strength) : pos(pos), strength(strength) { }
  };

private:

  Eigen::Vector2d pos, vel;
  double rot, rot_vel;

  double mass, mom_inertia;

  double restitution, mu_s, mu_k;

  std::vector<collider> colliders;
  double bounding_radius;

public:

  void apply_impulse (Vector2d position, Vector2d impulse) {
    vel += impulse / mass;
    rot_vel += position.cross (impulse) / mom_inertia;
  }

  bool rigid_body::process_collisions (const double restitution,
                                       const std::vector<Vector2d>& colliders_dpos_dtheta,
                                       const Vector2d& new_pos,
                                       const double new_rot,
                                       std::vector<bool>& collided);

};




#endif
