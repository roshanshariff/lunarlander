import numpy as np

from .framework import Framework

def initialize_worker(simulator, agent, agent_state=None):
    global framework
    np.random.seed()
    framework = Framework (simulator, agent)
    if agent_state:
        agent.set_state(np.frombuffer(agent_state))
    return framework

def run_episode(n):
    try:
        ep_return = framework.run_episode()
        if n == 0 or (n+1) % 100 == 0:
            print('Episode {}, return = {}'.format(n+1, ep_return))
            return (n, ep_return)
    except KeyboardInterrupt:
        pass
