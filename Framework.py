import numpy as np
import math

class Framework:

    def __init__ (self, simulator, agent, learn=True):
        self.simulator = simulator
        self.agent = agent
        self.learn = learn
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
        pos_x = 30*(2*np.random.random()-1)
        pos_y = 3.0 + 17*np.random.random()
        rot = math.radians(45)*(2*np.random.random()-1)

        self.simulator.initialize (pos_x=pos_x, pos_y=pos_y, rot=rot)

    def run (self, dt=float('inf')):

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
                if not self.learn: reward = None

                terminal = self.simulator.crashed or self.simulator.landed
                self.agent.update (reward, terminal)
                if terminal:
                    self.initialized = False
                    return False

        return True

    def reward (self):
        reward = -0.01 * self.simulator.main_throttle()
        if self.simulator.crashed or self.simulator.landed:
            reward -= abs(self.simulator.lander.pos[0]) / self.simulator.lander_width
        if self.simulator.crashed:
            reward -= 1.0 + math.log10(self.simulator.lander.breakage)
        return reward

    # def reward (self): 
    #     reward = -0.01*self.simulator.main_throttle()
    #     if self.simulator.crashed:
    #         breakage = self.simulator.breakage - 1.0
    #         reward += math.exp(-breakage**2) - 2.0
    #     elif self.simulator.landed:
    #         target_distance = abs(self.simulator.lander.pos[0]) / (2*self.simulator.lander_width)
    #         reward += 1.0 * math.exp(-target_distance**2)
    #     return reward
