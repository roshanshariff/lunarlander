import numpy as np

from Framework import Framework
from Simulator import LunarLanderSimulator
from PolicyGradientAgent import PolicyGradientAgent

simulator = LunarLanderSimulator()

def make_framework (Lambda):
    agent = PolicyGradientAgent (simulator, Lambda=Lambda)
    filename = 'data/saved_state_lambda'+str(Lambda)+'.npy'
    agent.save_state (filename)
    agent.load_state (filename, mmap_mode='r+')
    return Framework (simulator, agent)

lambdas = [ 0.0, 0.5, 0.75, 0.9 ]
frameworks = [ make_framework(Lambda) for Lambda in lambdas ] 
returns = [ np.frombuffer(f.returns) for f in frameworks ]

for (i, f) in enumerate(frameworks):
    print 'Lambda =', lambdas[i]
    f.train()

np.save ('data/lambdastudy_returns.npy', np.array(returns))

def make_plot ():
    for (i,r) in enumerate(returns):
        plot (r.cumsum(), label='$\lambda='+str(lambdas[i])+'$')
    legend (loc='lower left')
