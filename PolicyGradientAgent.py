import math
import numpy as np
import collections
import ctypes
import multiprocessing as mp
import scipy.weave as weave
import scipy.stats as stats

from TileCoder import TileCoder, HashingTileCoder

class PolicyGradientAgent:

    def __init__(self, simulator, dt=0.5, Lambda=0.75, alpha_v=0.1, alpha_u=0.1, num_features=2**20, tile_weight_exponent=0.5, 
                 trunc_normal=True, subspaces=[1,2,6]):

        self.simulator = simulator
        self.dt = max(dt, self.simulator.dt)

        self.tile_coder = HashingTileCoder (self.make_tile_coder(tile_weight_exponent, subspaces), num_features)

        initial_thrust_sigma = simulator.max_thrust / 10
        initial_thrust_mu = 0.5
        initial_rcs_sigma = simulator.max_rcs / 6
        initial_rcs_mu = 0.0

        self.critic = Critic (self.tile_coder, Lambda, alpha_v, initial_value=1.0)

        self.thrust_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_u, 
                                                 min_action=0.0, max_action=simulator.max_thrust, 
                                                 min_sigma=simulator.max_thrust/64, max_sigma=simulator.max_thrust/2,
                                                 initial_mu=initial_thrust_mu, initial_sigma=initial_thrust_sigma, 
                                                 trunc_normal=trunc_normal)

        self.rcs_actor = PolicyGradientActor (self.tile_coder, Lambda, alpha_u, 
                                              min_action=-simulator.max_rcs, max_action=simulator.max_rcs,
                                              min_sigma=simulator.max_rcs/32, max_sigma=simulator.max_rcs,
                                              initial_mu=initial_rcs_mu, initial_sigma=initial_rcs_sigma,
                                              trunc_normal=trunc_normal)

    def make_tile_coder (self, tile_weight_exponent, subspaces):
        #                            xpos   ypos  xvel  yvel        rot     rotvel
        state_signed  = np.array ([ False, False, True, True,      True,      True ])
        state_bounded = np.array ([  True,  True, True, True,     False,      True ])
        tile_size     = np.array ([    5.,    5.,   2.,   2., math.pi/2, math.pi/6 ])
        num_tiles     = np.array ([     6,     4,    4,    4,         2,         3 ])
        num_offsets   = np.array ([     2,     2,    4,    4,         8,         4 ])

        self.max_state = (tile_size * num_tiles) - 1e-8

        self.min_state = -self.max_state
        self.min_state[np.logical_not(state_signed)] = 0.0

        self.max_clip_state = self.max_state.copy()
        self.max_clip_state[np.logical_not(state_bounded)] = float('inf')

        self.min_clip_state = -self.max_clip_state
        self.min_clip_state[np.logical_not(state_signed)] = 0.0

        num_tiles[state_signed] *= 2
        num_tiles[state_bounded] += 1

        return TileCoder (tile_size, num_tiles, num_offsets, subspaces, tile_weight_exponent)

    def compute_action (self, features):

        # def clamp (value, low, high):
        #     value = low + math.fmod (abs(value-low), 2*(high-low))
        #     if value > high: value = 2*high - value
        #     return value

        # thrust = clamp (self.thrust_actor.act(features), 0.0, self.simulator.max_thrust)
        # rcs = clamp (self.rcs_actor.act(features), -self.simulator.max_rcs, self.simulator.max_rcs)
        thrust = self.thrust_actor.act(features)
        rcs = self.rcs_actor.act(features)
        return (thrust, rcs)

    def initialize (self, state):

        features = self.tile_coder.indices (state.clip (self.min_clip_state, self.max_clip_state))

        self.critic.initialize(features)
        self.thrust_actor.initialize()
        self.rcs_actor.initialize()

        return self.compute_action (features)

    def update (self, state, reward, terminal=False, learn=True):

        features = self.tile_coder.indices (state.clip (self.min_clip_state, self.max_clip_state))

        if learn:
            td_error = self.critic.evaluate (features, reward, terminal)
            self.thrust_actor.learn (td_error)
            self.rcs_actor.learn (td_error)

        return self.compute_action (features)

    def get_state(self):
        return np.vstack ((self.critic.value.weights,
                           self.thrust_actor.mu.weights,
                           self.thrust_actor.sigma.weights,
                           self.rcs_actor.mu.weights,
                           self.rcs_actor.sigma.weights))

    def set_state(self, state):
        state.shape = (5, self.tile_coder.num_features)
        (self.critic.value.weights,
         self.thrust_actor.mu.weights,
         self.thrust_actor.sigma.weights,
         self.rcs_actor.mu.weights,
         self.rcs_actor.sigma.weights) = state

    def save_state (self, savefile='data/saved_state.npy'):
        np.save (savefile, self.get_state())

    def load_state (self, savefile='data/saved_state.npy', mmap_mode=None):
        state = np.array (np.load (savefile, mmap_mode), copy=False)
        self.set_state(state)

    def persist_state(self, savefile=None, readonly=False):
        if savefile == None:
            state = np.frombuffer(mp.RawArray(ctypes.c_double, 5*self.tile_coder.num_features))
            state[:] = self.get_state().flat
            self.set_state(state)
        else:
            if not readonly: self.save_state(savefile)
            self.load_state (savefile, mmap_mode='r' if readonly else 'r+')

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

    def __init__ (self, tile_coder, Lambda, alpha, min_action, max_action, min_sigma, max_sigma, initial_mu, initial_sigma, gamma=1.0, 
                  trunc_normal=True):

        self.alpha = alpha
        self.min_action = min_action
        self.max_action = max_action
        self.min_sigma = min_sigma
        self.sigma_range = max_sigma - min_sigma

        initial_sigma_value = math.log((initial_sigma-self.min_sigma)/(max_sigma-initial_sigma)) 

        self.mu = LinearFunctionApprox (tile_coder, gamma*Lambda, initial_mu)
        self.sigma = LinearFunctionApprox (tile_coder, gamma*Lambda, initial_sigma_value)
        
        self.trunc_normal = trunc_normal

    def initialize (self):
        self.mu.initialize()
        self.sigma.initialize()

    def action_dist(self):

        mu = self.mu.value (self.features)
        sigma = self.sigma.value (self.features)
        sigma = self.min_sigma + self.sigma_range * (1 + math.tanh(sigma/2)) / 2

        max_mu = self.max_action + 3.0*sigma
        min_mu = self.min_action - 3.0*sigma
        mu = min (max (min_mu, mu), max_mu)

        if not self.trunc_normal: return (float('-inf'), float('inf'), mu, sigma)

        alpha = (self.min_action - mu) / sigma
        beta = (self.max_action - mu) / sigma
        return (alpha, beta, mu, sigma)

    def act(self, features):
        self.features = features
        if self.trunc_normal: 
            self.action = stats.truncnorm.rvs(*self.action_dist())
        else:
            self.action = np.random.normal(*self.action_dist()[2:])
        return self.action

    def learn(self, td_error):
        (alpha, beta, mu, sigma) = self.action_dist()
        std_action = (self.action - mu) / sigma

        if self.trunc_normal:
            trunc_weight = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
            trunc_grad_mu = (stats.norm.pdf(beta) - stats.norm.pdf(alpha)) / trunc_weight
            trunc_grad_sigma = (beta*stats.norm.pdf(beta) - alpha*stats.norm.pdf(alpha)) / trunc_weight
            if math.isnan(trunc_grad_sigma): trunc_grad_sigma = 0.0
        else:
            trunc_grad_mu = 0.0
            trunc_grad_sigma = 0.0

        variance = sigma**2 * (1 - trunc_grad_sigma - trunc_grad_mu**2)
        scaled_alpha = self.alpha * variance

        mu_grad = (std_action + trunc_grad_mu) / sigma
        self.mu.add_features(self.features, mu_grad)
        self.mu.update (scaled_alpha*td_error)

        sigma_grad = (std_action**2 - 1 + trunc_grad_sigma)/sigma*(sigma-self.min_sigma)*(1-(sigma-self.min_sigma)/self.sigma_range)

        self.sigma.add_features(self.features, sigma_grad)
        self.sigma.update (scaled_alpha*td_error)

    # def action_dist (self):
    #      action_mean = self.action_mean.value (self.features)
    #      action_stdev = self.action_stdev.value (self.features)
    #      action_stdev = self.max_stdev * (1 + math.tanh(action_stdev/2)) / 2
    #      return (action_mean, action_stdev)

    # def old_act (self, features):
    #     self.features = features
    #     self.action = np.random.normal(*self.action_dist())
    #     return self.action
        
    # def old_learn (self, td_error):

    #     (action_mean, action_stdev) = self.action_dist()
    #     std_action = (self.action - action_mean) / action_stdev

    #     action_mean_gradient = std_action / action_stdev
    #     self.action_mean.add_features (self.features, action_mean_gradient)

    #     #action_stdev_gradient = self.max_stdev*(std_action**2 - 1)*(1-action_stdev)
    #     action_stdev_gradient = (std_action**2 - 1)*(1 - action_stdev/self.max_stdev)
    #     self.action_stdev.add_features (self.features, action_stdev_gradient)

    #     scaled_alpha = self.alpha * action_stdev**2
    #     self.action_mean.update (scaled_alpha*td_error)
    #     self.action_stdev.update (scaled_alpha*td_error)

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

    # def value (self, features):
    #     return np.dot (self.weights.take(features), self.feature_weights)

    def value (self, features):
        feature_weights = self.feature_weights
        weights = self.weights
        code = """
            double value = 0.0;
            for (int i = 0; i < features.size(); ++i) {
                value += weights(features(i)) * feature_weights(i);
            }
            return_val = value; 
        """
        names = [ 'features', 'feature_weights', 'weights' ]
        return weave.inline (code, names, type_converters=weave.converters.blitz)

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

