import numpy as np
import math


class Collider:

    def __init__ (self, pos, normal_dir, strength, shear):
        self.pos = np.array(pos, dtype=np.float64)
        self.mu_s = float(mu_s)
        self.mu_k = float(mu_k)
        cos_dir = math.cos(normal_dir)
        sin_dir = math.sin(normal_dir)
        self.strength = np.matrix([[ cos_dir/strength, sin_dir/strength],
                                   [-sin_dir/shear,    cos_dir/shear   ]])        


class RigidBody:

    def __init__ (self, mass, mom_inertia, mu_s=0.0, mu_k=0.0, restitution=0.0, colliders=[]):

        self.pos = np.zeros (2)
        self.vel = np.zeros (2)
        self.rot = 0.0
        self.rot_vel = 0.0

        self.mass = float(mass)
        self.mom_inertia = float(mom_inertia)
        self.restitution = float(restitution)
        self.mu_s = float(mu_s)
        self.mu_k = float(mu_k)

        num_colliders = len(colliders)
        self.colliders = np.array (colliders, dtype=[('pos', np.float64, 2), ('strength', np.float64, (2,2))])
        self.bounding_radius = max ([ np.linalg.norm(c['pos']) for c in self.colliders ])

        self.collided = np.zeros (num_colliders, dtype=np.bool)
        self.contacted = np.zeros (num_colliders, dtype=np.bool)
        self.impulses = np.zeros_like (self.colliders['pos'])
        self.breakage = 0.0

    def apply_impulse (self, position, impulse):
        self.vel += impulse / self.mass
        self.rot_vel += np.cross(position, impulse) / self.mom_inertia

    def update (self, dt, force=np.zeros(2), torque=0.0):

        sin_rot = math.sin(self.rot)
        cos_rot = math.cos(self.rot)
        colliders_dpos_dtheta = np.dot([[-sin_rot, -cos_rot], [cos_rot, -sin_rot]],
                                       self.colliders['pos'].T).T

        def process_collision (restitution):

            new_pos = self.pos + dt*(self.vel + dt*force/self.mass)
            new_rot = self.rot + dt*(self.rot_vel + dt*torque/self.mom_inertia)

            collided = np.zeros_like (self.colliders, dtype=np.bool)
            if new_pos[1] > self.bounding_radius: return collided

            sin_new_rot = math.sin(new_rot)
            cos_new_rot = math.cos(new_rot)
            colliders_rel_pos = np.dot([[cos_new_rot, -sin_new_rot], [sin_new_rot, cos_new_rot]],
                                       self.colliders['pos'].T).T

            for i in np.argsort(colliders_rel_pos[:,1]):

                if new_pos[1] + colliders_rel_pos[i,1] > 0: break

                collider_vel = self.vel + self.rot_vel*colliders_dpos_dtheta[i]
                if collider_vel[1] > 0: continue

                K = np.array([colliders_rel_pos[i,1], -colliders_rel_pos[i,0]])
                K = np.outer(K,K)
                K /= self.mom_inertia
                K += np.eye(2)/self.mass

                impulse = np.linalg.solve(K, [-collider_vel[0], -(1+restitution)*collider_vel[1]])

                if abs(impulse[0]) > self.mu_s*impulse[1]:
                    friction = self.mu_k
                    if collider_vel[0] > 0: friction = -friction
                    impulse[1] = -(1+restitution) * collider_vel[1] / (friction*K[1,0] + K[1,1])
                    impulse[0] = impulse[1] * friction

                self.apply_impulse (colliders_rel_pos[i], impulse)
                collided[i] = True

                self.impulses[i] += np.dot([[cos_new_rot, sin_new_rot], [-sin_new_rot, cos_new_rot]],
                                           impulse)

            return collided

        # Collision
        self.impulses.fill(0.0)
        self.collided.fill(False)
        for i in xrange(10):
            collided = process_collision(self.restitution)
            if np.any(collided): self.collided |= collided
            else: break

        # Velocity update
        self.vel += dt * force/self.mass
        force.fill(0.0)
        self.rot_vel += dt * torque/self.mom_inertia
        torque = 0.0

        # Contact
        self.contacted.fill(False)
        for i in xrange(10):
            contacted = process_collision((i-9)/10.0)
            if np.any(contacted): self.contacted |= contacted
            else: break
        
        # Position update
        self.pos += dt * self.vel
        self.rot += dt * self.rot_vel

        # Breakage
        for i in xrange(self.colliders.size):
            breakage = np.linalg.norm (np.dot (self.colliders[i]['strength'], self.impulses[i]))
            self.breakage = max (self.breakage, breakage)

class LunarLanderSimulator:

    def __init__ (self, dt=1.0/20):

        self.dt = dt
        self.gravity = 1.622 # m/s^2

        self.lander_width = 9.07 # m

        # http://www.ibiblio.org/apollo/NARA-SW/R-567-sec6.pdf
        # Assuming descent stage 50% fuel, ascent stage 100% fuel
        mass = 11036.4 # kg
        mom_inertia = 28258.7 # kg m^2
        restitution = 0.2
        mu_s = 1.0
        mu_k = 0.9

        rcs_img_pos_y = 0.5508 # Y coordinate of RCS thrusters in image
        rcs_ref_pos_y = 6.4516 # Y coordinate of RCS thrusters in LM coords
        center_pos_y = 4.7515  # Y coordinate of Center of Mass in LM coords

        dps_force = 45040.8 # Force provided by descent engine
        rcs_force = 444.82 # Force provided by each RCS thruster
        num_rcs = 6 # number of RCS thrusters that can torque in one direction

        self.max_thrust = dps_force / mass # m/s^2
        self.max_rcs = num_rcs * rcs_force * (rcs_ref_pos_y - center_pos_y) / mom_inertia # rad/s^2

        self.image_center = np.array([0.5, rcs_img_pos_y+(center_pos_y-rcs_ref_pos_y)/self.lander_width])

        def image_coords (x, y):
            return ([x, y] - self.image_center) * self.lander_width

        def leg (x, y, strut_dir):
            strength = 3.0e4
            shear = strength * 4/10
            cos_dir = math.cos(strut_dir)
            sin_dir = math.sin(strut_dir)
            strength_mat = np.matrix([[ cos_dir/strength, sin_dir/strength],
                                      [-sin_dir/shear,    cos_dir/shear   ]])        
            return (image_coords(x,y), strength_mat)

        def body (x, y):
            return (image_coords(x,y), np.eye(2))
        
        self.lander = RigidBody (mass, mom_inertia, mu_s, mu_k, restitution,
                                 [ leg  (0.0541, 0.0456, math.pi/6),   # left leg bottom
                                   leg  (0.9459, 0.0456, math.pi*5/6), # right leg bottom
                                   leg  (0.0000, 0.0627, math.pi/6),   # left leg outer
                                   leg  (1.0000, 0.0626, math.pi*5/6), # right leg outer
                                   body (0.2251, 0.6980),
                                   body (0.4729, 0.8348),
                                   body (0.6211, 0.6809),
                                   body (0.7493, 0.4929) ])

        self.initialize()

        self.thruster_pos = image_coords(0.5078, 0.0899)
        self.thruster_radius = 0.5
        self.thruster_spread = 0.15

        self.rcs_pos_left = image_coords(0.2617, rcs_img_pos_y)
        self.rcs_pos_right = image_coords(0.7031, rcs_img_pos_y)
        self.rcs_radius = 0.25
        self.rcs_spread = 0.1

    def initialize (self, pos_x=0.0, pos_y=0.0, vel_x=0.0, vel_y=0.0, rot=0.0, rot_vel=0.0):

        min_pos_y = -np.min(np.dot([math.sin(rot), math.cos(rot)], self.lander.colliders['pos'].T))

        self.lander.pos[0] = pos_x
        self.lander.pos[1] = max(pos_y, min_pos_y)
        self.lander.rot = float(rot)

        self.lander.vel[0] = vel_x
        self.lander.vel[1] = vel_y
        self.lander.rot_vel = float(rot_vel)

        self.lander.breakage = 0.0
        self.crashed = False
        self.thrust = 0.0
        self.rcs = 0.0
        self.update()

    def update (self):

        accel = np.array([-self.thrust*math.sin(self.lander.rot),
                           self.thrust*math.cos(self.lander.rot) - self.gravity])

        self.lander.update(self.dt, accel*self.lander.mass, self.rcs*self.lander.mom_inertia)
        self.crashed |= self.lander.breakage > 1.0
        self.landed = np.all(self.lander.contacted[:2]) and np.linalg.norm(self.lander.vel) < 0.1

        #print self.lander.impulses[:,0] + self.lander.impulses[:,1]
        #if self.crashed: print self.lander.breakage
        
    def set_action (self, thrust, rcs):
        self.thrust = max (0, min (self.max_thrust, thrust))
        self.rcs = max (-self.max_rcs, min (self.max_rcs, rcs))

    def main_throttle (self):
        return self.thrust / self.max_thrust

    def rcs_throttle (self):
        return self.rcs / self.max_rcs

