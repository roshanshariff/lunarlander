import multiprocessing as mp
import numpy as np
import sys

def print_random (seed):
    np.random.seed(seed)
    print np.random.randint(10000)

def test_processes (num_processes=mp.cpu_count()):

    pool = mp.Pool(num_processes)
    results = []

    for i in range(num_processes):
        results.append (pool.apply_async (print_random, [np.random.randint(sys.maxint)]))

    for r in results: r.get()

    pool.close()
    pool.join()

test_processes()
