#!/usr/bin/env python

from os import path,makedirs
import numpy as np
import multiprocessing as mp

from lunarlander.framework import Framework
from lunarlander.simulator import LunarLanderSimulator
from lunarlander.policygradientagent import PolicyGradientAgent

import matplotlib.pyplot as plt

def run_experiment(num_episodes, Lambda, alpha, twe, trunc_normal, subspaces):

    np.random.seed()
    
    simulator = LunarLanderSimulator()
    agent = PolicyGradientAgent (simulator,
                                 Lambda=Lambda, alpha_u=alpha, alpha_v=alpha,
                                 tile_weight_exponent=twe,
                                 trunc_normal=trunc_normal,
                                 subspaces=subspaces)
    framework = Framework(simulator, agent)

    return np.array([framework.run_episode() for _ in range(num_episodes)])


def run_experiments(experiments):
    ctx = mp.get_context('spawn')
    with ctx.Pool() as pool:
        promises = {name: [pool.apply_async(run_experiment, (ex['num_episodes'],), ex['params'])
                           for _ in range(ex['num_runs'])]
                    for (name, ex) in experiments.items()}
        results = {name: np.vstack([p.get() for p in ps]) for (name, ps) in promises.items()}
    return results
        

def make_plot(results):
    for (name, returns) in results.items():
        p = experiments[name]['params']
        label = '\lambda={}, \alpha={}'.format(p['Lambda'], p['alpha'])
        plt.plot(returns.mean(axis=0).cumsum(), label=label)
    plt.legend (loc='lower left')
    plt.show()

experiments = {
    'weighted_trunc_normal': {
        'params': {'Lambda':0.75, 'alpha':0.1, 'twe':0.5, 'trunc_normal':True, 'subspaces':[1,2,6]},
        'num_runs':3, 'num_episodes':20000
    },
    'lambda_0.5_weighted_trunc_normal': {
        'params': {'Lambda':0.5, 'alpha':0.1, 'twe':0.5, 'trunc_normal':True, 'subspaces':[1,2,6]},
        'num_runs':3, 'num_episodes':20000
    },
    'lambda_0.9_weighted_trunc_normal': {
        'params': {'Lambda':0.9, 'alpha':0.1, 'twe':0.5, 'trunc_normal':True, 'subspaces':[1,2,6]},
        'num_runs':3, 'num_episodes':20000
    }
}

if __name__ == "__main__":
    results = run_experiments(experiments)
    np.savez_compressed('data/experiment.npz', kwds=results)
    make_plot(results)
