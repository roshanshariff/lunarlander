#include "simulator.hpp"

#include <utility>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#include <Eigen/Geometry>


bool rigid_body::process_collisions (const double restitution,
                                     const std::vector<Vector2d>& colliders_dpos_dtheta,
                                     const Vector2d& new_pos,
                                     const double new_rot,
                                     const bool processing_contact) {

  if (new_pos.y() > bounding_radius) return false;

  Eigen::Rotation2Dd rot_matrix (new_rot);

  std::vector<Vector2d> colliders_rel_pos (colliders.size());
  for (int i = 0; i < colliders.size(); ++i) {
    colliders_rel_pos[i] = rot_matrix * colliders[i].pos;
  }

  std::vector<std::pair<double, size_t> > colliders_sort_ix (colliders.size());
  for (int i = 0; i < colliders.size(); ++i) {
    colliders_sort_ix[i].first = colliders_rel_pos[i].y();
    colliders_sort_ix[i].second = i;
  }

  std::sort (colliders_sort_ix.begin(), colliders_sort_ix.end());

  bool collisions = false;

  for (int sorted_index = 0; sorted_index < colliders.size(); ++sorted_index) {

    const int i = colliders_sort_ix[sorted_index].second;

    if (new_pos.y() + colliders_rel_pos[i].y() > 0) break;

    const Vector2d collider_vel = vel + rot_vel*colliders_dpos_dtheta[i];
    if (collider_vel.y() > 0) continue;

    Matrix2d K;
    {
      Vector2d tangential_vel (colliders_rel_pos[i].y(), -colliders_rel_pos[i].x());
      K = tangential_vel * tangential_vel.transpose();
      K /= mom_inertia;
      K += Matrix2d::Identity() / mass;
    }

    Vector2d impulse = K.llt().solve (Vector2d (collider_vel.x(), -(1+restitution)*collider_vel.y()));

    if (std::abs (impulse.x()) > mu_s*impulse.y()) {
      const double friction = collider_vel.x() > 0 ? -mu_k : mu_k;
      impulse.y() = -(1+restitution) * collider_vel.y() / (friction*K(1,0) + K(1,1));
      impulse.x() = impulse.y() * friction;
    }

    apply_impulse (colliders_rel_pos[i], impulse);
    (processing_contact ? colliders[i].contacted : colliders[i].collided) = true;
    collisions = true;

    colliders[i].impulses += rot_matrix.inverse() * impulse;
  }

  return collisions;

}


void rigid_body::update (const double dt, const Vector2d& force, const double torque) {

  std::vector<Vector2d> colliders_dpos_dtheta (colliders.size());
  {
    Matrix2d rot_matrix;
    double sin_rot = std::sin (rot);
    double cos_rot = std::cos (rot);
    rot_matrix <<
      -sin_rot, -cos_rot,
      cos_rot,  -sin_rot;

    for (int i = 0; i < colliders.size(); ++i) {
      colliders_dpos_dtheta[i] = rot_matrix * colliders[i].pos;
    }
  }

  const Vector2d delta_vel = force * (dt / mass);
  const double delta_rot_vel = torque * (dt / mom_inertia);

  // Collision

  for (int i = 0; i < colliders.size(); ++i) colliders[i].reset_collision();

  for (int i = 0; i < 5; ++i) {

    const Vector2d new_pos = pos + dt*(vel + delta_vel);
    const double new_rot = rot + dt*(rot + delta_rot_vel);

    if (!process_collisions (restitution, colliders_dpos_dtheta, new_pos, new_rot, false)) break;
  }

  // Velocity update

  vel += delta_vel;
  rot_vel += delta_rot_vel;

  // Contact

  for (int i = -9; i <= 0; ++i) {

    const Vector2d new_pos = pos + dt*(vel + delta_vel);
    const double new_rot = rot + dt*(rot + delta_rot_vel);

    if (!process_collisions (i/10.0, colliders_dpos_dtheta, new_pos, new_rot, true)) break;
  }

  // Position update

  pos += dt * vel;
  rot += dt * rot_vel;

  // Breakage

  for (int i = 0; i < colliders.size(); ++i) {
    set_breakage ((colliders[i].strength * colliders[i].impulses).norm());
  }

}
