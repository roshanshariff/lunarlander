#!/usr/bin/env python

import numpy as np
import os
import sys

from Framework import Framework
from Simulator import LunarLanderSimulator
from PolicyGradientAgent import PolicyGradientAgent

simulator = LunarLanderSimulator()

def run_experiment(Lambda, alpha, twe, trunc_normal, subspaces, num_runs,num_episodes=20000, num_procs=None,name=""):
    returns = np.empty((num_runs, num_episodes), dtype=np.float64)
    results.append(returns)
    for i in xrange(num_runs):
        print name
        agent = PolicyGradientAgent (simulator, 
                                     Lambda=Lambda, alpha_u=alpha, alpha_v=alpha,
                                     tile_weight_exponent=twe,
                                     trunc_normal=trunc_normal,
                                     subspaces=subspaces)
        agent.persist_state()
        framework = Framework(simulator, agent, num_episodes=num_episodes)
        framework.train(num_procs=num_procs)
        returns[i] = framework.returns
    random = np.random.randint(sys.maxsize)

    directory = 'data/%s/' % (name)
    filename = directory + ('%d.npy' % (random))
    try:
        os.makedirs(directory)
    except OSError:
        pass
    np.save (filename, returns)
    return returns

def make_plot():
    for i,returns in enumerate(results):
        plot(returns.mean(axis=0).cumsum(), label=r'$\lambda=%g,\alpha=%g,p=%g$'%params[i][:3])
    legend (loc='lower left')

params = [
    {'Lambda':0, 'alpha':0.1, 'twe':0.5, 'trunc_normal':True, 'subspaces':[1,2,6], 'num_runs':1,
     'num_episodes':20000, 'num_procs':2, 'name':"lambda_0_weighted_trunc_normal"}
    {'Lambda':0.5, 'alpha':0.1, 'twe':0.5, 'trunc_normal':True, 'subspaces':[1,2,6], 'num_runs':1,
     'num_episodes':20000, 'num_procs':2, 'name':"lambda_0.5_weighted_trunc_normal"}
    {'Lambda':0.9, 'alpha':0.1, 'twe':0.5, 'trunc_normal':True, 'subspaces':[1,2,6], 'num_runs':1,
     'num_episodes':20000, 'num_procs':2, 'name':"lambda_0.9_weighted_trunc_normal"}
    ]

if __name__ == "__main__":
    results = []
    for ps in params:
        run_experiment(**ps)
    
