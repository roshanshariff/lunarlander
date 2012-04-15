import numpy as np
import multiprocessing as mp
import sys

from Framework import Framework
from Simulator import LunarLanderSimulator
from PolicyGradientAgent import PolicyGradientAgent

simulator = LunarLanderSimulator(dt=1.0/30)
agent = PolicyGradientAgent (simulator, dt=0.5)
framework = Framework (simulator, agent)

def manual_control ():
    import Display
    user_agent = Display.UserAgent(simulator)
    window = Display.LunarLanderWindow (Framework (simulator, user_agent))

def visualize ():
    import Display
    window = Display.LunarLanderWindow (framework)

def do_training (num_episodes, seed=None):
    if seed != None: np.random.seed (seed)
    for i in xrange(num_episodes):
        framework.run()
        print 'Return =', framework.Return

def mp_training (num_episodes, num_processes=mp.cpu_count()):

    pool = mp.Pool(num_processes)
    results = []

    while num_episodes > 0:
        batch_size = 1 + (num_episodes-1)/num_processes
        results.append (pool.apply_async (do_training, [batch_size, np.random.randint(sys.maxint)]))
        num_episodes -= batch_size
        num_processes -= 1

    for r in results: r.get()
    pool.close()
    pool.join()
