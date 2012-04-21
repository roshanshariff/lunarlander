import math
import numpy as np
import collections
import scipy.weave as weave

from TileCoder import TileCoder, HashingTileCoder

class PolicyGradientAgent:

    def __init__(self, simulator, dt=0.5, Lambda=0.75, alpha_v=0.1, alpha_u=0.1, num_features=2**20):

        self.simulator = simulator
        self.dt = max(dt, self.simulator.dt)

        self.tile_coder = HashingTileCoder (self.make_tile_coder(), num_features)

        # TODO Set max_stdev in a more principled way

        self.critic = Critic (self.tile_coder, Lambda, alpha_v, initial_value=1.0)

        self.thrust_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_u, max_stdev=simulator.max_thrust/4,
                                                 initial_mean=1.0, initial_stdev=simulator.max_thrust/8)

        self.rcs_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_u, max_stdev=simulator.max_rcs/2,
                                              initial_mean=0.0, initial_stdev=simulator.max_rcs/4)

        self.initialize()

    def make_tile_coder (self):

        state_signed  = np.array ([ False, False, True, True,      True,      True ])
        state_bounded = np.array ([  True,  True, True, True,     False,      True ])
        tile_size     = np.array ([    5.,    5.,   2.,   2., math.pi/2, math.pi/6 ])
        num_tiles     = np.array ([     6,     4,    4,    4,         2,         3 ])
        num_offsets   = np.array ([     2,     2,    4,    4,         8,         4 ])

        self.max_state = (tile_size * num_tiles) - 1e-10

        self.min_state = -self.max_state
        self.min_state[np.logical_not(state_signed)] = 0.0

        self.max_clip_state = self.max_state.copy()
        self.max_clip_state[np.logical_not(state_bounded)] = float('inf')

        self.min_clip_state = -self.max_clip_state
        self.min_clip_state[np.logical_not(state_signed)] = 0.0

        num_tiles[state_signed] *= 2
        num_tiles[state_bounded] += 1

        return TileCoder (tile_size, num_tiles, num_offsets, [0,1,2,6])

    def features (self):

        (pos_x, pos_y) = self.simulator.lander.pos
        (vel_x, vel_y) = self.simulator.lander.vel
        rot = self.simulator.lander.rot
        rot_vel = self.simulator.lander.rot_vel

        if pos_x < 0: (xsign, pos_x, vel_x, rot, rot_vel) = (-1.0, -pos_x, -vel_x, -rot, -rot_vel)
        else: xsign = 1.0

        state = np.array([pos_x, pos_y, vel_x, vel_y, rot, rot_vel]).clip(self.min_clip_state, self.max_clip_state)
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

        self.critic.initialize(features)
        self.thrust_actor.initialize()
        self.rcs_actor.initialize()

        self.take_action(features, xsign)

    def update (self, reward=None, terminal=False):

        (features, xsign) = self.features()

        if reward != None:
            td_error = self.critic.evaluate (features, reward, terminal)
            self.thrust_actor.learn (td_error)
            self.rcs_actor.learn (td_error)

        if not terminal:
            self.take_action (features, xsign)

    def save_state (self, savefile='data/saved_state.npy'):
        np.save (savefile, np.vstack ((self.critic.value.weights,
                                       self.thrust_actor.action_mean.weights,
                                       self.thrust_actor.action_stdev.weights,
                                       self.rcs_actor.action_mean.weights,
                                       self.rcs_actor.action_stdev.weights)))

    def load_state (self, savefile='data/saved_state.npy', mmap_mode=None):
        state = np.array (np.load (savefile, mmap_mode), copy=False)
        (self.critic.value.weights,
         self.thrust_actor.action_mean.weights,
         self.thrust_actor.action_stdev.weights,
         self.rcs_actor.action_mean.weights,
         self.rcs_actor.action_stdev.weights) = state

class Critic:

    def __init__ (self, tile_coder, Lambda, alpha, initial_value=0.0, gamma=1.0):

        self.alpha = alpha
        self.gamma = gamma

        self.value = LinearFunctionApprox (tile_coder, gamma*Lambda, initial_value)

    def initialize (self, features):
        self.features = features
        self.value.initialize()

    def evaluate (self, new_features, reward, terminal=False):

        old_value = self.value.value (self.features)
        new_value = self.value.value (new_features) if not terminal else 0.0
        td_error = reward + self.gamma*new_value - old_value

        self.value.add_features (self.features)
        self.value.update (self.alpha*td_error)

        self.features = new_features
        return td_error

class PolicyGradientActor:

    def __init__ (self, tile_coder, Lambda, alpha, max_stdev, initial_mean, initial_stdev, gamma=1.0):

        self.alpha = alpha
        self.max_stdev = max_stdev

        initial_stdev_value = math.log(initial_stdev/(max_stdev-initial_stdev))

        self.action_mean = LinearFunctionApprox (tile_coder, gamma*Lambda, initial_mean)
        self.action_stdev = LinearFunctionApprox (tile_coder, gamma*Lambda, initial_stdev_value)

    def initialize (self):
        self.action_mean.initialize()
        self.action_stdev.initialize()

    def action_dist (self):
        action_mean = self.action_mean.value (self.features)
        action_stdev = self.action_stdev.value (self.features)
        action_stdev = self.max_stdev * (1 + math.tanh(action_stdev/2)) / 2
        return (action_mean, action_stdev)

    def act (self, features):
        self.features = features
        self.action = np.random.normal(*self.action_dist())
        return self.action
        
    def learn (self, td_error):

        (action_mean, action_stdev) = self.action_dist()
        std_action = (self.action - action_mean) / action_stdev

        action_mean_gradient = std_action / action_stdev
        self.action_mean.add_features (self.features, action_mean_gradient)

        action_stdev_gradient = self.max_stdev*(std_action**2 - 1)*(1-action_stdev)
        self.action_stdev.add_features (self.features, action_stdev_gradient)

        scaled_alpha = self.alpha * action_stdev**2
        self.action_mean.update (scaled_alpha*td_error)
        self.action_stdev.update (scaled_alpha*td_error)

class LinearFunctionApprox:

    def __init__ (self, tile_coder, falloff, initial_value=0.0, threshold=0.05):
        self.feature_weights = tile_coder.feature_weights
        self.weights = np.empty (tile_coder.num_features)
        self.weights.fill (initial_value/self.feature_weights.sum())
        self.falloff = falloff

        trace_len = int(math.ceil(math.log(threshold, falloff))) if falloff > 0 else 1
        self.trace = collections.deque (maxlen=trace_len)

    def initialize (self):
        self.trace.clear()

    def value (self, features):
        return np.dot (self.weights.take(features), self.feature_weights)

    def add_features (self, features, scaling=1.0):
        self.trace.appendleft ((features, scaling))

    def update (self, step_size):
        feature_weights = self.feature_weights
        weights = self.weights
        code = """
            double amount = double(step_size) * double(scaling);
            for (int i = 0; i < features.size(); ++i) {
                weights(features(i)) += amount * feature_weights(i);
            }
        """
        names = [ 'step_size', 'feature_weights', 'weights', 'features', 'scaling' ]
        falloff = self.falloff
        for (features, scaling) in self.trace:
            weave.inline (code, names, type_converters=weave.converters.blitz)
            step_size *= falloff


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

