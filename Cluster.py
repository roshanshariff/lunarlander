from cStringIO import StringIO
import math
import numpy as np
from os import path
import subprocess
import sys

def experiment(num_runs, episodes, Lambda, alpha_v, alpha_u, trunc_normal, 
               tile_weight_exponent):
    
    executable_path = path.join(path.dirname(path.abspath(__file__)), "build", "main")
    arguments = map(str,[executable_path,
                         "--lambda", Lambda,
                         "--alpha-v", alpha_v,
                         "--alpha-u", alpha_u,
                         "--episodes", episodes,
                         "--trunc-normal" if trunc_normal else "--no-trunc-normal",
                         "--tile-weight-exponent", tile_weight_exponent])
    print arguments
    
    results = np.empty((num_runs, episodes))

    for i in xrange(num_runs):
        p = subprocess.Popen(arguments, stdout=subprocess.PIPE)
        results[i,:] = np.loadtxt(p.stdout)
        p.wait()

    return results.tolist()

