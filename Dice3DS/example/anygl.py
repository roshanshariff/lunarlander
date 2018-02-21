# anygl.py

"""Import the appropriate OpenGL wrapper.

The package used to wrap OpenGL can be configured by setting
Dice3DS.example.config.OPENGL_PACKAGE.  The wrappers supported are
PyOpenGL and pyglet.

All the symbols in the OpenGL wrapper package will be imported into
the module's namespace.  In addition, there are a few extra symbols
defined so that code written for PyOpenGL can be made compatible with
pyglet.gl.

"""

from Dice3DS.example import config

if config.OPENGL_PACKAGE == "pyglet":

    from pyglet.gl import *
    import ctypes

    # Define functions to adapt the arguments to the ctypes buffers
    # that pyglet.gl expects

    def gl_float_array(*args):
        return (ctypes.c_float*len(args))(*args)

    ctypes.pythonapi.PyObject_AsReadBuffer.argtypes = [
        ctypes.py_object,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t) ]
    ctypes.pythonapi.PyObject_AsReadBuffer.restype = None

    def gl_pointer(buf):
        b = ctypes.c_void_p()
        i = ctypes.c_size_t()
        ctypes.pythonapi.PyObject_AsReadBuffer(
            buf,ctypes.byref(b),ctypes.byref(i))
        return b

    # There are some other small differences to make pyglet.gl and
    # PyOpenGL incompatible. Patch them up here.

    _pyglet_glGenTextures = glGenTextures

    def glGenTextures(n):
        v = ctypes.c_ulong()
        _pyglet_glGenTextures(n,ctypes.byref(v))
        return v.value


elif config.OPENGL_PACKAGE == "PyOpenGL":

    from OpenGL.GL import *
    from OpenGL.GLU import *

    def gl_float_array(*args):
        return args

    def gl_pointer(buf):
        return buf

else:
    raise ValueError('Dice3DS.example.config.OPENGL_PACKAGE '
                     'is set to an invalid value ("%.200r")'
                     % config.OPENGL_PACKAGE)
