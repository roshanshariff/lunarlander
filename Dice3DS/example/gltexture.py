# gltexture.py

"""OpenGL texture object abstraction.

Provides a class that is an abstraction of OpenGL texture objects. It
can create textures from image files, and automatically generates
mipmaps if requested.

"""

from builtins import object
from Dice3DS.example.anygl import *
from Dice3DS.example import anyimgload


def _create_texture_from_image(imagedata, wrap_s, wrap_t,
                               magfilter, minfilter):
    if imagedata.format == 'RGBX':
        iformat = GL_RGB
    elif imagedata.format == 'RGBA':
        iformat = GL_RGBA
    else:
        raise ValueError('only RGBX and RGBA supported')

    mipmap = minfilter in (GL_NEAREST_MIPMAP_NEAREST,
                           GL_NEAREST_MIPMAP_LINEAR,
                           GL_LINEAR_MIPMAP_NEAREST,
                           GL_LINEAR_MIPMAP_LINEAR)

    ti = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D,ti)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,wrap_s)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,wrap_t)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,magfilter)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,minfilter)

    if mipmap:
        gluBuild2DMipmaps(GL_TEXTURE_2D, iformat, imagedata.width,
                          imagedata.height, GL_RGBA, GL_UNSIGNED_BYTE,
                          imagedata.raw_data)
    else:
        glTexImage2D(GL_TEXTURE_2D, 0, iformat, imagedata.width,
                     imagedata.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     imagedata.raw_data)

    return ti, (imagedata.width,imagedata.height)


class Texture(object):
    """An OpenGL texture object.

    This class uses PIL to create and bind an OpenGL texture
    object.

    Features:

        It automatically creates mipmaps if a mipmap rendering
        option is selected.

        Handles alpha channel in images.

    Methods:

        tex.enable() - enable OpenGL texturing of proper dimensions
        tex.disable() - disable OpenGL texturing of proper dimensions
        tex.bind() - bind this texture
        tex.real() - whether this is really a texture (which it is)
        tex.destroy() - delete OpenGL texture object

    """

    def __init__(self, filespec, wrap_s=GL_REPEAT, wrap_t=GL_REPEAT,
                 magfilter=GL_LINEAR, minfilter=GL_LINEAR):
        """Create a GL texture object.

            tex = Texture(filename, wrap_s=GL_REPEAT, wrap_s=GL_REPEAT,
                    magfilter=GL_LINEAR, minfilter=GL_LINEAR)

        filespec is one of three options for loading a file:
            - a filename
            - a tuple (zip_file_object, filename_within_zipfile)
            - an open file stream (for reading)
        
        wrap_s, wrap_t, magfilter, and minfilter are some
        OpenGL texture options.

        """

        imagedata = anyimgload.load_image(filespec)
        self.index, self.size = _create_texture_from_image(
            imagedata,wrap_s,wrap_t,magfilter,minfilter)

    def real(self):
        """Return True if this is really a texture"""
        return True

    def enable(self):
        """Enable OpenGL texturing of the proper dimensions."""
        glEnable(GL_TEXTURE_2D)

    def disable(self):
        """Disable OpenGL texturing of the proper dimensions."""
        glDisable(GL_TEXTURE_2D)

    def bind(self):
        """Bind this texture."""
        glBindTexture(GL_TEXTURE_2D, self.index)

    def destroy(self):
        """Destroy this texture; release OpenGL texture object."""
        glDeleteTextures((self.index,))
        del self.index



class NonTexture(object):
    """An OpenGL non texture object.

    This is just a sort of null class indicating an object has no
    texture.  It provides the same methods the Texture class does,
    but they do nothing.

    """
    
    def real(self):
        return False
    def enable(self):
        pass
    def disable(self):
        pass
    def bind(self):
        pass
    def destroy(self):
        pass
