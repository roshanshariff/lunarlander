import numpy as np
import multiprocessing as mp
import ctypes
import math
import sys

class Framework:

    def __init__ (self, simulator, agent):
        self.simulator = simulator
        self.agent = agent
        self.initialize()

    def initialize (self):
        self.initialize_simulator()
        self.simulator_time = self.simulator.dt
        self.agent.initialize()
        self.agent_time = self.agent.dt
        self.Return = 0.0
        self.initialized = True

    def initialize_simulator (self):

        # pos_x = -2*self.simulator.lander_width
        # pos_y = 15.0
        # rot = math.radians(45)*(2*np.random.random()-1)

        state = np.random.random (self.agent.max_state.size)
        state *= self.agent.max_state - self.agent.min_state
        state += self.agent.min_state
        state[2:6] = 0.0

        if np.random.randint(0,2): state[0] = -state[0]
        state[4] = np.random.normal (0.0, math.pi/8)
        # state[5] = np.random.normal (0.0, math.pi/16)

        self.simulator.initialize (*state)

    def run (self, dt=float('inf'), learn=True):

        if not self.initialized: self.initialize()

        while dt > 0.0:

            elapsed_time = min (self.simulator_time, self.agent_time, dt)
            self.simulator_time -= elapsed_time
            self.agent_time -= elapsed_time
            dt -= elapsed_time

            if self.simulator_time <= 0.0:

                self.simulator_time += self.simulator.dt
                self.simulator.update()

            if self.agent_time <= 0.0:

                self.agent_time += self.agent.dt

                reward = self.reward()
                self.Return += reward
                if not learn: reward = None

                terminal = self.simulator.crashed or self.simulator.landed
                self.agent.update (reward, terminal)
                if terminal:
                    self.initialized = False
                    return False

        return True

    def reward (self):
        reward = 0.0
        if self.simulator.crashed or self.simulator.landed:
            reward -= abs(self.simulator.lander.pos[0]) / self.simulator.lander_width
            if self.simulator.crashed:
                reward -= 1.0
        if not self.simulator.landed:
            reward -= math.log10 (1.0 + self.simulator.lander.breakage)
            self.simulator.lander.breakage = 0.0
        #if reward != 0.0: print 'Reward =', reward
        reward -= 0.01 * self.simulator.main_throttle()
        return reward

    def train (self, num_episodes, time_limit=600.0, num_procs=mp.cpu_count()):
        lock = mp.Lock()
        counter = mp.RawValue(ctypes.c_uint, 0)
        def proc (seed):
            np.random.seed (seed)
            while True:
                with lock:
                    i = int(counter.value)
                    counter.value += 1
                if i < num_episodes:
                    if self.run(dt=time_limit):
                        print '%d: Return = %g (time limit exceeded)' % (i, self.Return)
                        self.initialized = False
                    else:
                        print '%d: Return = %g' % (i, self.Return)
                else:
                    break
        procs = [ mp.Process (target=proc, args=(np.random.randint(sys.maxint),)) for i in xrange(num_procs) ]
        try:
            for p in procs: p.start()
            for p in procs: p.join()
        finally:
            for p in procs: p.terminate()
