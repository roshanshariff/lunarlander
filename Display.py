import math
import pyglet
from pyglet.gl import *
from pyglet.window import key
import numpy as np

try:
    from lepton import Particle, ParticleGroup
    from lepton.domain import Plane, Cylinder, Sphere
    from lepton.controller import Lifetime, Movement, Bounce, Collector
    from lepton.renderer import BillboardRenderer
    from lepton.emitter import StaticEmitter
    from lepton.texturizer import SpriteTexturizer
    lepton_loaded = True
except ImportError:
    lepton_loaded = False
    print 'Lepton not found; particle effects disabled.'


class LunarLanderWindow (pyglet.window.Window):

    def __init__ (self, framework, fov_diag=math.radians(46.8)):

        super(LunarLanderWindow, self).__init__(resizable=True, visible=False)

        self.framework = framework
        
        self.push_handlers (framework.agent)

        self.simulator = framework.simulator
        self.fov_diag = fov_diag
        self.scaleFactor = 100

        self.load_resources()

        self.set_caption('Lunar Lander')
        glDisable(GL_DEPTH_TEST)

        self.set_visible (True)
        self.start()
        pyglet.app.run()
        pyglet.clock.unschedule (self.update)

    def load_resources (self):

        lander_width = self.scaleFactor * self.simulator.lander_width

        lander_img = pyglet.resource.image('resources/lander.png')
        lander_img.anchor_x = lander_img.width*self.simulator.image_center[0]
        lander_img.anchor_y = lander_img.height*self.simulator.image_center[1]
        self.lander = pyglet.sprite.Sprite(lander_img)
        self.lander.scale = lander_width/lander_img.width

        shadow_img = pyglet.resource.image('resources/shadow-texture.png')
        shadow_img.anchor_x = shadow_img.width * 0.5
        shadow_img.anchor_y = shadow_img.height * 0.5
        self.shadow = pyglet.sprite.Sprite(shadow_img, x=0, y=0)
        self.shadow.scale = 5*lander_width/shadow_img.width

        target_img = pyglet.resource.image('resources/target.png')
        target_img.anchor_x = target_img.width * 0.5
        target_img.anchor_y = target_img.height * 0.5
        self.target = pyglet.sprite.Sprite(target_img, x=0, y=0)
        self.target.scale = 2.0*lander_width/target_img.width

        ground_img = pyglet.resource.image('resources/moon_light.jpg')
        self.ground_tex = ground_img.get_image_data().get_mipmapped_texture()

        self.crashed = pyglet.text.Label (text='CRASH', font_size=100, bold=True,
                                          color=(255,0,0,255), anchor_x='center', anchor_y='top')
        self.crashed.visible = False

        self.landed = pyglet.text.Label (text='LANDED', font_size=100, bold=True,
                                         color=(255,255,255,255), anchor_x='center', anchor_y='top')
        self.landed.visible = False

        if lepton_loaded:
            self.puff_texture = pyglet.resource.texture('resources/puff.png')
            ground_domain = Plane ([0,0,0], [0,1,0], half_space=True)
            self.thruster = ParticleGroup (
                controllers=[Lifetime(0.3), Movement(), Bounce(ground_domain, 0.01, -5), Collector(ground_domain)],
                renderer=BillboardRenderer(SpriteTexturizer(self.puff_texture.id)))
            self.rcs = ParticleGroup (
                controllers=[Lifetime(0.1), Movement(), Collector(ground_domain)],
                renderer=BillboardRenderer(SpriteTexturizer(self.puff_texture.id)))

    def start (self, wait=0.0):
        dt = self.simulator.dt
        pyglet.clock.unschedule (self.update)
        pyglet.clock.schedule_once (lambda _: pyglet.clock.schedule_interval (self.update, dt), wait)

    def update (self, _):

        dt = self.simulator.dt
        if not self.framework.run(dt, learn=False):
            print 'Return =', self.framework.Return
            self.start(1.0)

        self.update_particles(dt)

        (self.lander.x, self.lander.y) = self.scaleFactor*self.simulator.lander.pos
        self.lander.rotation = -math.degrees(self.simulator.lander.rot)

        self.shadow.x = self.lander.x

        self.crashed.visible = self.simulator.crashed
        self.landed.visible = self.simulator.landed

    def update_particles (self, dt):

        if not lepton_loaded: return

        self.thruster.update(dt)
        self.rcs.update(dt)

        cos_rot = math.cos(self.simulator.lander.rot)
        sin_rot = math.sin(self.simulator.lander.rot)
        rot_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

        def exhaust_particles (group, opacity, vel, pos, radius, spread):
            
            if opacity == 0: return

            pos = self.simulator.lander.pos + np.dot(rot_matrix, pos)
            vel = self.simulator.lander.vel + np.dot(rot_matrix, vel)
            spread *= self.scaleFactor * np.linalg.norm(vel)

            pos_domain = Cylinder(list(self.scaleFactor*pos)+[0.0],
                                  list(self.scaleFactor*(pos-vel*dt))+[0.0],
                                  self.scaleFactor*radius)
            vel_domain = Sphere(list(self.scaleFactor*vel)+[0.0], spread)

            size = self.scaleFactor * 2.0 * radius
            emitter = StaticEmitter (template=Particle(color=(1,1,1,0.1*opacity), size=(size,size), age=-dt),
                                     position=pos_domain, velocity=vel_domain)
            emitter.emit(int(2000*dt), group)

        thrust = self.simulator.main_throttle()
        exhaust_particles (self.thruster, thrust, [0,-50], self.simulator.thruster_pos,
                           self.simulator.thruster_radius, self.simulator.thruster_spread)

        rcs = self.simulator.rcs_throttle()
        if rcs > 0:
            exhaust_particles (self.rcs, rcs, [50,0], self.simulator.rcs_pos_right,
                               self.simulator.rcs_radius, self.simulator.rcs_spread)
        else:
            exhaust_particles (self.rcs, -rcs, [-50,0], self.simulator.rcs_pos_left,
                               self.simulator.rcs_radius, self.simulator.rcs_spread)


    def on_draw (self):
        self.clear()

        self.push_camera_projection()
        self.push_default_view()

        self.push_ground_view()
        self.draw_ground_plane()
        self.target.draw()
        self.shadow.draw()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        if lepton_loaded:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            self.thruster.draw()
            self.rcs.draw()

        self.lander.draw()
        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        if self.crashed.visible: self.crashed.draw()
        if self.landed.visible: self.landed.draw()

    def push_camera_projection (self):

        lander_width = self.scaleFactor * self.simulator.lander_width
        lander_height = self.scaleFactor * -self.simulator.lander.colliders['pos'][1].min()

        (pad_x, pad_y) = (0.0, lander_height)
        (lander_x, lander_y) = self.scaleFactor * self.simulator.lander.pos

        self.display_width = abs(pad_x - lander_x) + 2*lander_width
        self.display_height = abs(pad_y - lander_y) + 2*lander_width

        if self.display_width*self.height > self.display_height*self.width:
            self.display_height = self.display_width * self.height / self.width
        else:
            self.display_width = self.display_height * self.width / self.height

        self.camera_x = (pad_x + lander_x) / 2.0
        self.camera_y = (pad_y + lander_y) / 2.0
        self.camera_z = self.display_height / (2.0 * math.tan(self.fov_y/2.0))

        moon_radius = self.scaleFactor * 1.7371e6
        self.camera_near = self.camera_y / math.tan(self.fov_y/2.0)
        self.camera_far = math.sqrt(2*moon_radius*self.camera_y)

        glMatrixMode (GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective (math.degrees(self.fov_y), self.display_width/self.display_height,
                        self.camera_near, self.camera_far)

    def push_default_view (self):
        glMatrixMode (GL_MODELVIEW)
        glPushMatrix()
        gluLookAt (self.camera_x, self.camera_y, self.camera_z,
                   self.camera_x, self.camera_y, 0.0,
                   0.0, 1.0, 0.0)

    def push_ground_view (self):
        glMatrixMode (GL_MODELVIEW)
        glPushMatrix()
        gluLookAt(0, 0, 0, 0, -1, 0, 0, 0, -1)

    def draw_ground_plane (self):
        
        znear = self.camera_z - self.camera_near
        zfar = self.camera_z - self.camera_far
        wnear = self.camera_near * math.tan(self.fov_x/2.0)
        wfar = self.camera_far * math.tan(self.fov_x/2.0)
        x = self.camera_x

        coords = (x-wnear, znear, x+wnear, znear, x+wfar, zfar, x-wfar, zfar)

        glEnable (self.ground_tex.target)
        glBindTexture (self.ground_tex.target, self.ground_tex.id)
        glMatrixMode(GL_TEXTURE)
        glPushMatrix()
        glLoadIdentity()
        glTranslated(0.5, 0.5, 0)
        glScaled(1.0/(200*self.scaleFactor), 1.0/(200*self.scaleFactor), 0.0)
        pyglet.graphics.draw (4, GL_QUADS, ('v2f', coords), ('t2f', coords))
        glPopMatrix()
        glBindTexture (self.ground_tex.target, 0)
        glDisable (self.ground_tex.target)

    def on_resize (self, width, height):

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)

        diag = 2.0 * math.tan(self.fov_diag/2.0)/math.hypot(width,height)
        self.fov_x = 2.0 * math.atan (width*diag/2.0)
        self.fov_y = 2.0 * math.atan (height*diag/2.0)
        self.crashed.x=self.width/2.0
        self.crashed.y=self.height
        self.landed.x=self.width/2.0
        self.landed.y=self.height

class UserAgent (pyglet.window.key.KeyStateHandler):

    def __init__ (self, simulator):
        super(UserAgent, self).__init__()
        self.simulator = simulator
        self.dt = 0.5 #simulator.dt
        self.max_state = np.array([30.0,20.0,5.0,5.0,1000,5])
        self.min_state = np.array([0.0,0.0,-5.0,-5.0,1000,-5])

    def initialize (self, state):
        pass

    def update (self, state, reward, terminal=False, learn=False):
        thrust = self.simulator.max_thrust if self[key.W] else 0.0
        left = self.simulator.max_rcs if self[key.A] else 0.0
        right = self.simulator.max_rcs if self[key.D] else 0.0
        self.simulator.set_action(thrust, left-right)

