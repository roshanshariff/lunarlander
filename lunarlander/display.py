from __future__ import (division, print_function)

import math
import pyglet
import pyglet.gl as gl
from pyglet.window import key, FPSDisplay
import numpy as np
import ctypes
import os

import Dice3DS
import Dice3DS.example.config
Dice3DS.example.config.OPENGL_PACKAGE = "PyOpenGL"
Dice3DS.example.config.IMAGE_LOAD_PACKAGE = "pyglet"
from Dice3DS.example import gltexture, glmodel


class LunarLanderWindow (pyglet.window.Window):

    def __init__ (self, framework, fov_diag=math.radians(46.8)):

        super(LunarLanderWindow, self).__init__(resizable=True, visible=False,
                                                config=gl.Config(double_buffer=True, depth_size=24, stencil_size=1))

        self.framework = framework
        self.push_handlers (framework.agent)
        self.simulator = framework.simulator
        self.dt = self.simulator.dt

        self.fov_diag = fov_diag
        self.scaleFactor = 100

        self.load_resources()

        self.set_caption('Lunar Lander')
        self.fps_display = FPSDisplay(self)

        self.set_visible (True)
        self.start()
        pyglet.app.run()
        pyglet.clock.unschedule (self.update)

    def load_resources (self):

        pyglet.resource.path = ['@{}.resources'.format(__package__)]
        pyglet.resource.reindex()

        lander_width = self.scaleFactor * self.simulator.lander_width

        # lander_img = pyglet.resource.image('lander.png')
        # lander_img.anchor_x = lander_img.width*self.simulator.image_center[0]
        # lander_img.anchor_y = lander_img.height*self.simulator.image_center[1]
        # self.lander = pyglet.sprite.Sprite(lander_img)
        # self.lander.scale = lander_width/lander_img.width

        # shadow_img = pyglet.resource.image('shadow-texture.png')
        # shadow_img.anchor_x = shadow_img.width * 0.5
        # shadow_img.anchor_y = shadow_img.height * 0.5
        # self.shadow = pyglet.sprite.Sprite(shadow_img, x=0, y=0)
        # self.shadow.scale = 5*lander_width/shadow_img.width

        target_img = pyglet.resource.image('target.png')
        target_img.anchor_x = target_img.width * 0.5
        target_img.anchor_y = target_img.height * 0.5
        self.target = pyglet.sprite.Sprite(target_img, x=0, y=0)
        self.target.scale = 2.0*lander_width/target_img.width

        self.ground_tex = pyglet.resource.texture('moon_light.jpg').get_image_data().get_mipmapped_texture()

        self.crashed = pyglet.text.Label (text='CRASH', font_size=100, bold=True,
                                          color=(255, 0, 0, 255), anchor_x='center', anchor_y='top')
        self.landed = pyglet.text.Label (text='LANDED', font_size=100, bold=True,
                                         color=(255, 255, 255, 255), anchor_x='center', anchor_y='top')

        self.puff_texture = pyglet.resource.texture('puff.png')
        self.particles = Thruster.make_domain()
        self.thruster = Thruster (self.particles, self.scaleFactor, self.dt,
                                  np.append(self.simulator.thruster_pos, 0), self.simulator.thruster_radius,
                                  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T, 30, self.simulator.thruster_spread,
                                  2000, 0.5, 0.1)
        self.rcs_left = Thruster (self.particles, self.scaleFactor, self.dt,
                                  np.append(self.simulator.rcs_pos_left, 0), self.simulator.rcs_radius,
                                  np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T, 30, self.simulator.rcs_spread,
                                  1000, 0.3, 0.25)
        self.rcs_right = Thruster (self.particles, self.scaleFactor, self.dt,
                                   np.append(self.simulator.rcs_pos_right, 0), self.simulator.rcs_radius,
                                   np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T, 30, self.simulator.rcs_spread,
                                   1000, 0.3, 0.25)

        lem_model_dir = 'lunarlandernofoil_c'

        with pyglet.resource.file(os.path.join(lem_model_dir, 'lunarlandernofoil_carbajal.3ds')) as lem_model_file:
            lem_dom = Dice3DS.dom3ds.read_3ds_mem(lem_model_file.read())

        def load_lem_texture (filename):
            filename = os.path.splitext(filename.decode("utf-8"))[0] + '.png'
            with pyglet.resource.file(os.path.join(lem_model_dir, filename)) as tex:
                return gltexture.Texture(tex)

        self.lem_model = glmodel.GLModel (lem_dom, load_lem_texture,
                                          nfunc=Dice3DS.util.calculate_normals_by_cross_product)

        # for m in self.lem_model.meshes:
        #     if m.matarrays:
        #         for (material, faces) in m.matarrays:
        #             if material.texture:
        #                 print(material.texture.real)

    def start (self, wait=0.0):
        pyglet.clock.unschedule (self.update)
        pyglet.clock.schedule_once (lambda _: pyglet.clock.schedule_interval (self.update, self.dt), wait)

    def update (self, dt):

        # print('Updating: {} s'.format(dt))

        if dt > self.dt:
            dt = self.dt

        if not self.framework.run(dt, learn=False):
            self.start(1.0)

        self.update_particles()

        # (self.lander.x, self.lander.y) = self.scaleFactor*self.simulator.lander.pos
        # self.lander.rotation = -math.degrees(self.simulator.lander.rot)
        # self.shadow.x = self.lander.x

    def update_particles (self):

        pos = np.append (self.simulator.lander.pos, 0)
        vel = np.append (self.simulator.lander.vel, 0)
        cos_rot = math.cos(self.simulator.lander.rot)
        sin_rot = math.sin(self.simulator.lander.rot)
        rot = np.array([[cos_rot, -sin_rot, 0], [sin_rot, cos_rot, 0], [0, 0, 1]])
        rot_vel = np.array ([0, 0, self.simulator.lander.rot_vel])

        thrust = self.simulator.main_throttle()
        rcs = self.simulator.rcs_throttle()

        self.thruster.update (thrust, pos, vel, rot, rot_vel)
        self.rcs_left.update (max(0, -rcs), pos, vel, rot, rot_vel)
        self.rcs_right.update (max(0, rcs), pos, vel, rot, rot_vel)

    def on_draw (self):

        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)
        gl.glDepthMask(gl.GL_FALSE)

        with push_matrix(gl.GL_PROJECTION):
            self.set_camera_projection()

            with push_matrix(gl.GL_MODELVIEW):
                gl.glLoadIdentity()
                gl.gluLookAt (self.camera_x, self.camera_y, self.camera_z,
                              self.camera_x, self.camera_y, 0.0,
                              0.0, 1.0, 0.0)

                with push_matrix(gl.GL_MODELVIEW):
                    gl.gluLookAt(0, 0, 0, 0, -1, 0, 0, 0, -1)
                    self.draw_ground_plane()
                    self.target.draw()
                    # self.shadow.draw()

                self.draw_shadow_lem()
                self.draw_lem()

                with push_attrib(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT):
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
                    gl.glEnable(gl.GL_DEPTH_TEST)
                    with bound_texture (self.puff_texture.target, self.puff_texture.id):
                        Thruster.draw(self.particles)

        if self.simulator.crashed:
            self.crashed.draw()
        if self.simulator.landed:
            self.landed.draw()

        self.fps_display.draw()

    def draw_lem (self):
        with push_attrib(gl.GL_DEPTH_BUFFER_BIT | gl.GL_LIGHTING_BIT):
            gl.glDepthMask(gl.GL_TRUE)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_NORMALIZE)
            gl.glShadeModel(gl.GL_SMOOTH)
            gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, float_ctype_array(1.0, 1.0, 1.0, 1.0))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, float_ctype_array(0.0, 0.0, 0.0, 1.0))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, float_ctype_array(1.0, 1.0, 1.0, 1.0))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, float_ctype_array(0.5, 0.5, 0.5, 1.0))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, float_ctype_array(1.0, 1.0, 1.0, 0.0))
            self.draw_lem_model()

    def draw_shadow_lem (self):
        with push_attrib(gl.GL_STENCIL_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT):
            gl.glStencilFunc(gl.GL_NOTEQUAL, 1, 1)
            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_REPLACE)
            gl.glEnable(gl.GL_STENCIL_TEST)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_ZERO, gl.GL_CONSTANT_ALPHA)
            gl.glBlendColor(1.0, 1.0, 1.0, 0.3)
            with push_matrix(gl.GL_MODELVIEW):
                gl.glMultTransposeMatrixf(float_ctype_array(1.0, -1.0, 0.0, 0.0,
                                                            0.0, 0.0, 0.0, 0.0,
                                                            0.0, 0.0, 1.0, 0.0,
                                                            0.0, 0.0, 0.0, 1.0))
                self.draw_lem_model()

    def draw_lem_model (self):
        with push_matrix(gl.GL_MODELVIEW):
            gl.glTranslatef(*np.append(self.scaleFactor*self.simulator.lander.pos, 0))
            gl.glRotatef(-math.degrees(self.simulator.lander.rot), 0, 0, -1)
            gl.glScalef(*[self.scaleFactor]*3)

            gl.glTranslatef(0, -3, 0)
            gl.glRotatef(0, 0, 1, 0)
            gl.glRotatef(90, -1, 0, 0)
            gl.glScalef(*[2**0.5]*3)
            self.lem_model.render()

    def set_camera_projection (self):

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
        # self.camera_far = 500*self.scaleFactor
        self.camera_far = math.sqrt(2*moon_radius*self.camera_y)

        gl.glLoadIdentity()
        gl.gluPerspective (math.degrees(self.fov_y), self.display_width/self.display_height,
                           self.camera_near, self.camera_far)

    def draw_ground_plane (self):

        znear = self.camera_z - self.camera_near
        zfar = self.camera_z - self.camera_far
        wnear = self.camera_near * math.tan(self.fov_x/2.0)
        wfar = self.camera_far * math.tan(self.fov_x/2.0)
        x = self.camera_x

        coords = (x-wnear, znear, x+wnear, znear, x+wfar, zfar, x-wfar, zfar)

        with push_matrix(gl.GL_TEXTURE):
            gl.glLoadIdentity()
            gl.glTranslated(0.5, 0.5, 0)
            gl.glScaled(1.0/(200*self.scaleFactor), 1.0/(200*self.scaleFactor), 0.0)
            with bound_texture(self.ground_tex.target, self.ground_tex.id):
                pyglet.graphics.draw (4, gl.GL_QUADS, ('v2f', coords), ('t2f', coords))

    def on_resize (self, width, height):

        super(LunarLanderWindow, self).on_resize(width, height)

        diag = 2.0 * math.tan(self.fov_diag/2.0) / math.hypot(width, height)
        self.fov_x = 2.0 * math.atan (width*diag/2.0)
        self.fov_y = 2.0 * math.atan (height*diag/2.0)
        self.crashed.x = self.width/2.0
        self.crashed.y = self.height
        self.landed.x = self.width/2.0
        self.landed.y = self.height


class Thruster (object):

    tri_verts = (0, 1, 2, 2, 1, 3)

    @staticmethod
    def make_domain ():
        return pyglet.graphics.vertexdomain.create_domain('v3f/stream', 't2s/static', 'c4f/stream')

    @staticmethod
    def draw (domain):
        domain.draw(gl.GL_TRIANGLES)

    def __init__ (self, domain, scale, dt, pos, radius, rot, vel, vel_spread, rate, duration, opacity):

        self.scale = scale
        self.dt = dt

        self.thruster_pos = pos * scale
        self.thruster_radius = radius * scale
        self.thruster_rot = rot

        self.exhaust_speed = vel*scale*dt
        self.exhaust_vel = np.dot (rot, [0, 0, -self.exhaust_speed])
        self.exhaust_vel_stddev = vel_spread*self.exhaust_speed

        self.nparticles = int(math.ceil(dt * rate))
        self.nframes = int(math.ceil(duration / dt))

        self.positions = np.zeros((self.nframes, self.nparticles, 3), dtype=np.float32)
        self.velocities = np.zeros((self.nframes, self.nparticles, 3), dtype=np.float32)
        self.ixframe = 0

        self.vertices = domain.create(self.nframes*self.nparticles*6)

        self.init_tex_coords()
        self.init_vert_colors()
        self.init_vert_coords(2*self.thruster_radius)

        self.particle_opacity = opacity

    def init_tex_coords (self):
        coords = [(x, y) for y in (1, 0) for x in (0, 1)]
        tri_tex_coords = np.array([coords[i] for i in self.__class__.tri_verts], dtype=np.int16)
        tex_coords = np.tile (tri_tex_coords, (self.nframes, self.nparticles, 1, 1))
        self.vertices.tex_coords = numpy_to_ctype_array(tex_coords, gl.GLshort)

    def init_vert_colors (self):
        self.frame_colors = np.ones((self.nparticles, 6, 4), dtype=np.float32)
        self.vertices.colors = numpy_to_ctype_array(np.tile (self.frame_colors, (self.nframes, 1)), gl.GLfloat)

    def set_frame_opacity (self, frame, opacity):
        n = self.frame_colors.size
        self.frame_colors[..., 3] = opacity * self.particle_opacity
        self.vertices.colors[frame*n:(frame+1)*n] = numpy_to_ctype_array(self.frame_colors, gl.GLfloat)

    def init_vert_coords (self, size):
        coords = [(x/2.0, y/2.0, 0.0) for y in (-size, size) for x in (-size, size)]
        self.tri_vert_coords = np.array([coords[i] for i in self.__class__.tri_verts], dtype=np.float32)
        self.vert_coords = np.zeros((self.nframes, self.nparticles, 6, 3), dtype=np.float32)
        self.update_vert_coords()

    def update_vert_coords (self):
        np.add (self.positions[..., np.newaxis, :], self.tri_vert_coords, out=self.vert_coords)
        self.vertices.vertices = numpy_to_ctype_array(self.vert_coords, gl.GLfloat)

    def move_particles (self):
        self.positions += self.velocities
        collided = self.positions[..., 1] < 0
        self.positions[collided, 1] = 0
        self.velocities[collided, 0] *= 4.0
        self.velocities[collided, 1] *= -0.05
        self.velocities[collided, 2] *= 4.0

    def update (self, thrust, pos, vel, rot, rot_vel):

        self.move_particles()

        self.set_frame_opacity(self.ixframe, thrust)
        particle_pos = self.positions[self.ixframe]
        particle_vel = self.velocities[self.ixframe]
        self.ixframe = (self.ixframe + 1) % self.nframes

        pos = pos * self.scale
        vel = vel * self.scale * self.dt
        rot_vel = rot_vel * self.dt
        thruster_pos = np.dot(rot, self.thruster_pos)

        buffer0 = np.random.normal (scale=self.exhaust_vel_stddev, size=(self.nparticles, 3))
        buffer0 += vel + np.dot(rot, self.exhaust_vel) + np.cross(rot_vel, thruster_pos)
        particle_vel[:] = buffer0

        buffer1 = np.random.random_sample ((self.nparticles, 3))
        buffer1 *= [2*math.pi, self.thruster_radius**2, self.exhaust_speed]
        np.cos (buffer1[:, 0], out=buffer0[:, 0])
        np.sin (buffer1[:, 0], out=buffer0[:, 1])
        np.sqrt (buffer1[:, 1], out=buffer1[:, 1])
        buffer0[:, :2] *= buffer1[:, 1, np.newaxis]
        buffer0[:, 2] = buffer1[:, 2]

        np.dot (buffer0, np.dot(rot, self.thruster_rot).T, out=buffer1)
        buffer1 += pos + thruster_pos
        particle_pos[:] = buffer1

        self.update_vert_coords()


class UserAgent (pyglet.window.key.KeyStateHandler):

    def __init__ (self, simulator):
        super(UserAgent, self).__init__()
        self.simulator = simulator
        self.dt = simulator.dt
        self.max_state = np.array([30.0, 20.0, 5.0, 5.0, 1000, 5])
        self.min_state = np.array([0.0, 0.0, -5.0, -5.0, 1000, -5])
        self.Return = 0.0

    def initialize (self, state):
        return (0.0, 0.0)

    def update (self, state, reward, terminal=False, learn=False):
        thrust = self.simulator.max_thrust if self[key.W] else 0.0
        left = self.simulator.max_rcs if self[key.A] else 0.0
        right = self.simulator.max_rcs if self[key.D] else 0.0
        if self.simulator.lander.pos[0] < 0:
            (left, right) = (right, left)
        self.Return += reward
        if terminal:
            print ('Return = {}'.format(self.Return))
            self.Return = 0.0
        return (thrust, left-right)


def numpy_to_ctype_array (array, dtype):
    return array.ctypes.data_as(ctypes.POINTER(dtype*array.size)).contents


def float_ctype_array (*args):
    return (ctypes.c_float * len(args))(*args)


class push_matrix (object):

    def __init__ (self, matrix_mode):
        self.matrix_mode = matrix_mode

    def __enter__ (self):
        gl.glPushAttrib(gl.GL_TRANSFORM_BIT)
        gl.glMatrixMode(self.matrix_mode)
        gl.glPushMatrix()

    def __exit__ (self, type, value, traceback):
        gl.glMatrixMode(self.matrix_mode)
        gl.glPopMatrix()
        gl.glPopAttrib()


class push_attrib (object):

    def __init__ (self, mask):
        self.mask = mask

    def __enter__ (self):
        gl.glPushAttrib(self.mask)

    def __exit__ (self, type, value, traceback):
        gl.glPopAttrib()


class bound_texture (object):

    def __init__ (self, target, id):
        self.target = target
        self.id = id

    def __enter__ (self):
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_TEXTURE_BIT)
        gl.glEnable(self.target)
        gl.glBindTexture(self.target, self.id)

    def __exit__ (self, type, value, traceback):
        gl.glPopAttrib()
