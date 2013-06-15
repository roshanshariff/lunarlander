#include "simulator.hpp"

#include <utility>
#include <algorithm>
#include <cstdlib>

#include <Eigen/Geometry>

bool rigid_body::process_collisions (const double restitution,
                                     const std::vector<Vector2d>& colliders_dpos_dtheta,
                                     const Vector2d& new_pos,
                                     const double new_rot,
                                     std::vector<bool>& collided) {

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
    collided[i] |= true;
    collisions = true;

  }

  return collisions;

}
