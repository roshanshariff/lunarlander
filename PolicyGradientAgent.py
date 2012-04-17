import math
import numpy as np
import collections
import scipy.weave as weave

from TileCoder import RoshanTileCoder, HashingTileCoder

class PolicyGradientAgent:

    def __init__(self, simulator, dt, Lambda=0.9, alpha_v=0.1, alpha_u=0.1, num_features=2**20):

        self.simulator = simulator
        self.dt = max(dt, self.simulator.dt)

        self.tile_coder = HashingTileCoder (self.make_tile_coder(), num_features)

        # TODO Set max_stdev in a more principled way
        self.value_critic = Critic (self.tile_coder, Lambda, alpha_v, value=0.0)
        self.thrust_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_u, max_stdev=simulator.max_thrust/4, mean=1.0, stdev=simulator.max_thrust/8)
        self.rcs_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_v, max_stdev=simulator.max_rcs/2, mean=0.0, stdev=simulator.max_rcs/4)

        self.initialize()

    def make_tile_coder (self):

        state_signed  = np.array ([ False, False, True, True,      True,      True ])
        state_bounded = np.array ([  True,  True, True, True,     False,      True ])
        tile_size     = np.array ([    5.,    5.,   2.,   2., math.pi/2, math.pi/4 ])
        num_tiles     = np.array ([     6,     4,    4,    4,         2,         4 ])
        num_offsets   = np.array ([     2,     2,    4,    4,         8,         4 ])

        self.max_state = (tile_size * num_tiles) - 1e-10
        self.max_state[np.logical_not(state_bounded)] = float('inf')

        self.min_state = -self.max_state
        self.min_state[np.logical_not(state_signed)] = 0.0

        num_tiles[state_signed] *= 2
        num_tiles[state_bounded] += 1

        return RoshanTileCoder (tile_size, num_tiles, num_offsets, [0,1,2,6])

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

    def save_state (self, savefile='data/saved_state.npy'):
        np.save (savefile, np.vstack ((self.value_critic.value,
                                       self.thrust_actor.action_mean,
                                       self.thrust_actor.action_stdev,
                                       self.rcs_actor.action_mean,
                                       self.rcs_actor.action_stdev)))

    def load_state (self, savefile='data/saved_state.npy', mmap_mode=None):
        (self.value_critic.value,
         self.thrust_actor.action_mean,
         self.thrust_actor.action_stdev,
         self.rcs_actor.action_mean,
         self.rcs_actor.action_stdev) = np.load (savefile, mmap_mode)

class Critic:

    def __init__ (self, tile_coder, Lambda, alpha, value=0.0, gamma=1.0):

        self.alpha = alpha
        self.gamma = gamma
        self.feature_weights = tile_coder.feature_weights

        self.eligibility = DequeEligibilityTrace (self.feature_weights, gamma*Lambda)

        self.value = np.empty(tile_coder.num_features)
        self.value.fill (value)

    def initialize (self, features):
        self.features = features
        self.eligibility.clear()

    def evaluate (self, new_features, reward, terminal=False):

        old_value = np.dot(self.value[self.features], self.feature_weights)
        new_value = np.dot(self.value[new_features], self.feature_weights) if not terminal else 0.0
        td_error = reward + self.gamma*new_value - old_value

        self.eligibility.add_features(self.features)
        self.eligibility.add_to_vector(self.value, self.alpha*td_error)

        self.features = new_features
        return td_error

class PolicyGradientActor:

    def __init__ (self, tile_coder, Lambda, alpha, max_stdev, mean, stdev, gamma=1.0):

        self.alpha = alpha
        self.gamma = gamma
        self.max_stdev = max_stdev
        self.feature_weights = tile_coder.feature_weights

        self.eligibility_mean = DequeEligibilityTrace (self.feature_weights, gamma*Lambda)
        self.eligibility_stdev = DequeEligibilityTrace (self.feature_weights, gamma*Lambda)

        self.action_mean = np.empty (tile_coder.num_features)
        self.action_mean.fill (mean)

        self.action_stdev = np.empty (tile_coder.num_features)
        self.action_stdev.fill (-math.log((max_stdev-stdev)/stdev))

    def initialize (self):
        self.eligibility_mean.clear()
        self.eligibility_stdev.clear()

    def action_dist (self):
        action_mean = np.dot(self.action_mean[self.features], self.feature_weights)
        action_stdev = self.max_stdev / (1.0 + math.exp(-np.dot(self.action_stdev[self.features], self.feature_weights)))
        return (action_mean, action_stdev)

    def act (self, features):
        self.features = features
        self.action = np.random.normal(*self.action_dist())
        return self.action
        
    def learn (self, td_error):

        (action_mean, action_stdev) = self.action_dist()
        std_action = (self.action - action_mean) / action_stdev

        alpha = self.alpha * action_stdev**2

        action_mean_gradient = std_action / action_stdev
        self.eligibility_mean.add_features (self.features, action_mean_gradient)
        self.eligibility_mean.add_to_vector (self.action_mean, alpha*td_error)

        action_stdev_gradient = self.max_stdev*(std_action**2 - 1)*(1-action_stdev)
        self.eligibility_stdev.add_features (self.features, action_stdev_gradient)
        self.eligibility_stdev.add_to_vector (self.action_stdev, alpha*td_error)

class EligibilityTrace:

    def __init__ (self, feature_weights, falloff, threshold=0.05):
        self.falloff = falloff
        self.length = int(math.ceil(math.log(threshold, falloff)))
        self.weights = np.empty(self.length, dtype=np.float64)
        self.features = np.empty((self.length, feature_weights.size), dtype=np.intp)
        self.feature_weights = feature_weights
        self.clear()

    def add_features (self, features, weight=1.0):
        self.weights *= self.falloff
        self.weights[self.ix] = weight
        self.features[self.ix] = features
        self.ix = (self.ix + 1) % self.length

    def add_to_vector (self, vec, value):
        weights = self.weights
        features = self.features
        feature_weights = self.feature_weights
        length, num_features = features.shape
        weave.inline ("""
            for (int i = 0; i < length; i++) {
                double amount = double(weights(i)) * double(value);
                for (int j = 0; j < num_features; j++) {
                    vec(features(i,j)) += amount * double(feature_weights(j));
                }
            }
        """, locals().keys(), type_converters=weave.converters.blitz)

    def clear (self):
        self.weights.fill(0.0)
        self.features.fill(0)
        self.ix = 0

class DequeEligibilityTrace:

    def __init__ (self, feature_weights, falloff, threshold=0.05):
        self.feature_weights = feature_weights
        self.falloff = falloff
        self.trace = collections.deque (maxlen=int(math.ceil(math.log(threshold, falloff))))
        self.clear()

    def add_features (self, features, weight=1.0):
        self.trace.appendleft ((features, weight))

    def add_to_vector (self, vec, value):
        feature_weights = self.feature_weights
        falloff = self.falloff
        code = """
            double amount = double(weight) * double(value);
            for (int i = 0; i < feature_weights.size(); ++i) {
                vec(features(i)) += amount * feature_weights(i);
            }
        """
        names = [ 'vec', 'value', 'feature_weights', 'features', 'weight' ]
        for (features, weight) in self.trace:
            weave.inline (code, names, type_converters=weave.converters.blitz)
            value *= falloff

    def clear (self):
        self.trace.clear()
