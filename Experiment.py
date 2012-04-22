import numpy as np

from Framework import Framework
from Simulator import LunarLanderSimulator
from PolicyGradientAgent import PolicyGradientAgent

simulator = LunarLanderSimulator()

def run_experiment(Lambda, alpha, tile_weight_exponent, num_runs, num_episodes=200000, num_procs=None):
    returns = np.empty((num_runs, num_episodes), dtype=np.float64)
    for i in xrange(num_runs):
        print 'Lambda = %g, Alpha = %g, p = %g Run %d:'%(Lambda, alpha, tile_weight_exponent, i)
        agent = PolicyGradientAgent (simulator, 
                                     Lambda=Lambda, alpha_u=alpha, alpha_v=alpha,
                                     tile_weight_exponent=tile_weight_exponent)
        agent.persist_state()
        framework = Framework(simulator, agent, num_episodes=num_episodes)
        framework.train(num_procs=num_procs)
        returns[i] = framework.returns
    filename = 'data/experiment_%g_%g_%g.npy'%(Lambda,alpha,tile_weight_exponent)
    np.save (filename, returns)
    return returns

def make_plot():
    for i,returns in enumerate(results):
        plot(returns.mean(axis=0).cumsum(), label=r'$\lambda=%g,\alpha=%g,p=%g$'%params[i][:3])
    legend (loc='lower left')
        

if __name__ == "__main__":
    params = [ (0.75, 0.1, 0.5, 10) ]
    results = []
    for ps in params:
        results.append(run_experiment(*ps))
    
