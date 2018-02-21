# config.py

"""Configuration parameters for Dice3DS example libraries.

Defines variables that determine what package is used to perform
certain tasks.

   OPENGL_PACKAGE

       Determines which OpenGL wrapper package to use.
       Can be "PyOpenGL" or "pyglet".

   IMAGE_LOAD_PACKAGE

       Determines which package to use when loading images.
       Can be "PIL", "pyglet", or "pygame".

These value must be set before importing any other modules in the
example package; otherwise they have no effect.

"""

OPENGL_PACKAGE = "PyOpenGL"

IMAGE_LOAD_PACKAGE = "PIL"
