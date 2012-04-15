import math
import numpy as np
import scipy.sparse as sp
import random

from TileCoder import TileCoder, HashingTileCoder

class PolicyGradientAgent:

    def __init__(self, simulator, dt, Lambda=0.9, alpha_v=0.1, alpha_u=0.1, num_features=2**20):

        self.simulator = simulator
        self.dt = max(dt, self.simulator.dt)

        self.tile_coder = HashingTileCoder (self.make_tile_coder(), num_features)

        # TODO Set max_stdev in a more principled way
        self.value_critic = Critic (self.tile_coder, Lambda, alpha_v, value=0.0)
        self.thrust_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_u, max_stdev=1.0, mean=1.0, stdev=0.5)
        self.rcs_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_v, max_stdev=1.0, mean=0.0, stdev=0.5)

        self.initialize()

    def make_tile_coder (self):

        state_doubled = np.array([0, 0, 1, 1, 0, 1])
        state_bounded = np.array([1, 1, 1, 1, 0, 1])

        cell_size = np.array([5., 5., 5., 5., math.pi/2, math.pi/4])
        num_cells = np.array([6,  4,  4,   4,         4,         4])
        num_samples = np.array([2, 2, 4, 4, 4, 4])

        self.max_state = (cell_size * num_cells) - 1e-10
        self.min_state = -self.max_state*state_doubled
        self.max_state[4] = float('inf')
        self.min_state[4] = -float('inf')

        return TileCoder (cell_size, (num_cells*(1+state_doubled))+state_bounded, num_samples, [0,1,2,6])

    def features (self):

        (pos_x, pos_y) = self.simulator.lander.pos
        (vel_x, vel_y) = self.simulator.lander.vel
        rot = self.simulator.lander.rot
        rot_vel = self.simulator.lander.rot_vel

        if pos_x < 0: (xsign, pos_x, vel_x, rot, rot_vel) = (-1.0, -pos_x, -vel_x, -rot, -rot_vel)
        else: xsign = 1.0

        state = np.array([pos_x, pos_y, vel_x, vel_y, rot, rot_vel]).clip(self.min_state, self.max_state)
        return (self.tile_coder.indices(state), xsign)

    def take_action (self, features, xsign):

        def clamp (value, low, high):
            value = low + math.fmod (abs(value-low), 2*(high-low))
            if value > high: value = 2*high - value
            return value

        thrust = clamp (self.thrust_actor.act(features), 0.0, self.simulator.max_thrust)
        rcs = clamp (self.rcs_actor.act(features), -self.simulator.max_rcs, self.simulator.max_rcs)
        self.simulator.set_action (thrust, xsign*rcs)

    def initialize (self):

        (features, xsign) = self.features()

        self.value_critic.initialize(features)
        self.thrust_actor.initialize()
        self.rcs_actor.initialize()

        self.take_action(features, xsign)

    def update (self, reward=None, terminal=False):

        (features, xsign) = self.features()

        if reward != None:
            td_error = self.value_critic.evaluate (features, reward, terminal)
            self.thrust_actor.learn (td_error)
            self.rcs_actor.learn (td_error)

        if not terminal:
            self.take_action (features, xsign)

    def save_state (self, savefile='saved_state.npy'):
        np.save (savefile, np.vstack ((self.value_critic.value,
                                       self.thrust_actor.action_mean,
                                       self.thrust_actor.action_stdev,
                                       self.rcs_actor.action_mean,
                                       self.rcs_actor.action_stdev)))

    def load_state (self, savefile='saved_state.npy', mmap_mode=None):
        (self.value_critic.value,
         self.thrust_actor.action_mean,
         self.thrust_actor.action_stdev,
         self.rcs_actor.action_mean,
         self.rcs_actor.action_stdev) = np.load (savefile, mmap_mode)

class Critic:

    def __init__ (self, tile_coder, Lambda, alpha, value=0.0, gamma=1.0):

        self.alpha = alpha/tile_coder.active_features
        self.gamma = gamma

        self.eligibility = EligibilityTrace (tile_coder.active_features, gamma*Lambda)

        self.value = np.empty(tile_coder.num_features)
        self.value.fill (value / tile_coder.active_features)

    def initialize (self, features):
        self.features = features
        self.eligibility.clear()

    def evaluate (self, new_features, reward, terminal=False):

        value = np.sum(self.value[new_features]) if not terminal else 0.0
        td_error = reward + self.gamma*value - np.sum(self.value[self.features])

        self.eligibility.add_features(self.features)
        self.eligibility.add_to_vector(self.value, self.alpha*td_error)

        self.features = new_features
        return td_error

class PolicyGradientActor:

    def __init__ (self, tile_coder, Lambda, alpha, max_stdev, mean, stdev, gamma=1.0):

        self.alpha = alpha/tile_coder.active_features
        self.gamma = gamma
        self.max_stdev = max_stdev

        self.eligibility_mean = EligibilityTrace (tile_coder.active_features, gamma*Lambda)
        self.eligibility_stdev = EligibilityTrace (tile_coder.active_features, gamma*Lambda)

        self.action_mean = np.empty (tile_coder.num_features)
        self.action_mean.fill (mean / tile_coder.active_features)

        self.action_stdev = np.empty (tile_coder.num_features)
        self.action_stdev.fill (-math.log((max_stdev-stdev)/stdev) / tile_coder.active_features)

    def initialize (self):
        self.eligibility_mean.clear()
        self.eligibility_stdev.clear()

    def act (self, features):

        action_mean = np.sum(self.action_mean[features])
        action_stdev = self.max_stdev / (1.0 + math.exp(-np.sum(self.action_stdev[features])))

        self.action = random.gauss(action_mean, action_stdev)
        self.features = features

        return self.action
        
    def learn (self, td_error):

        action_mean = np.sum(self.action_mean[self.features])
        action_stdev = self.max_stdev / (1.0 + math.exp(-np.sum(self.action_stdev[self.features])))
        std_action = (self.action - action_mean) / action_stdev

        alpha = self.alpha * action_stdev**2

        action_mean_gradient = std_action / action_stdev
        self.eligibility_mean.add_features (self.features, action_mean_gradient)
        self.eligibility_mean.add_to_vector (self.action_mean, alpha*td_error)

        action_stdev_gradient = self.max_stdev*(std_action**2 - 1)*(1-action_stdev)
        self.eligibility_stdev.add_features (self.features, action_stdev_gradient)
        self.eligibility_stdev.add_to_vector (self.action_stdev, alpha*td_error)

class EligibilityTrace:

    def __init__ (self, active_features, falloff, threshold=0.01):
        self.falloff = falloff
        self.length = int(math.ceil(math.log(threshold, falloff)))
        self.weights = np.empty(self.length)
        self.features = np.empty((self.length, active_features), dtype=np.int)
        self.clear()

    def add_features (self, features, weight=1.0):
        self.weights *= self.falloff
        self.weights[self.ix] = weight
        self.features[self.ix,:] = features
        self.ix = (self.ix + 1) % self.length

    def add_to_vector (self, vec, value):
        for ix in range(self.length):
            vec[self.features[ix,:]] += self.weights[ix]*value

    def clear (self):
        self.weights.fill(0.0)
        self.features.fill(0)
        self.ix = 0

