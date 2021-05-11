import json
import sys

from lunarlander.simulator import LunarLanderSimulator
from lunarlander.display import LunarLanderWindow

class Framework:

    def __init__ (self, simulator, input):
        self.simulator = simulator
        self.input = input
        if self.read_input():
            self.agent = object()
            self.display = LunarLanderWindow (self)

    def run (self, dt, learn):
        self.read_input()
        return not (self.simulator.crashed or self.simulator.landed)

    def read_input (self):
        input = self.input.readline()
        if input:
            self.update (**json.loads (input))
            return True
        else:
            return False

    def update (self, Return, x, y, vx, vy, rot, vrot, thrust, rcs, breakage, crashed, landed):

        self.Return = Return

        self.simulator.lander.pos[0] = x
        self.simulator.lander.pos[1] = y
        self.simulator.lander.rot = float(rot)

        self.simulator.lander.vel[0] = vx
        self.simulator.lander.vel[1] = vy
        self.simulator.lander.rot_vel = float(vrot)

        self.simulator.thrust = thrust
        self.simulator.rcs = rcs

        self.simulator.lander.breakage = breakage
        self.simulator.crashed = crashed
        self.simulator.landed = landed

simulator = LunarLanderSimulator()
simulator.dt = 0.05
framework = Framework (simulator, sys.stdin)
