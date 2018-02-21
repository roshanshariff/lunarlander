# anyimgload.py

"""Define a function to load an image with the appropriate package.

The package used to load images can be configured by setting
Dice3DS.example.config.IMAGE_LOAD_PACKAGE.  The image loaders
supported are PIL (Python Imaging Library), pygame, and pyglet.

The function defined is called load_image().  It implements a file
search because 3DS files don't care about case or even filename
extensions.  It will upsize textures to the next power of 2 if
necessary.  It is zipfile aware (helpful since some 3DS models are
delivered as zips, although the stupid messages they cat to the end of
zip files can break this).  The output of this function is a class
containing image raw data in a format suitable for OpenGL textures,
format ("RGB" or "RGBA"), width, and height.

"""

from future import standard_library
standard_library.install_aliases()
from builtins import object
from Dice3DS.example import config

import io
import os
import sys


# Data class to store attributes of an image

class ImageData(object):
    """Holds data for a given image.

    Has four attributes:
        raw_data: the raw image data
        format: RGBA or RGBX
        width: image width in pixels
        height: image height in pixels

    """
    
    def __init__(self,raw_data,format,width,height):
        self.raw_data = raw_data
        self.format = format
        self.width = width
        self.height = height


# The load image function. Calls load_image_specific to do its dirty
# work, which depends on the configuration

def load_image(filespec):
    """Loads an image using the configured 

        image_data = load_image(filespec)

    filespec is one of three options for loading a file:
        - a filename
        - a tuple (zip_file_object, filename_within_zipfile)
        - an open file stream (for reading)

    Returns an ImageData() instance.

    The image loading library used will depend on the value of the
    configuration variable Dice3DS.example.config.IMAGE_LOAD_PACKAGE.

    """

    if hasattr(filespec,"read"):
        return load_image_specific(filespec)
    if isinstance(filespec,tuple):
        zfo,imgfilename = filespec
        flo,hint = search_for_file_zipfile(zfo,imgfilename)
        try:
            return load_image_specific(flo,hint)
        finally:
            flo.close()
    flo,hint = search_for_file_filesystem(filespec)
    try:
        return load_image_specific(flo,hint)
    finally:
        flo.close()


# The next two function perform an intelligent filename search to
# solve a stupid problem. Not only do actual filename cases not always
# match what's in 3DS, sometimes the extension doesn't either.  These
# check for the exact filename first, then search the directory for a
# filename differing only by case, then for a filename with a
# different extension.  One is for the filesystem, the other is for
# zip files.

def search_for_file_filesystem(imgfilename):
    try:
        return open(imgfilename,"rb"), imgfilename
    except IOError:
        imgdirname,imgbasename = os.path.split(imgfilename)
        if imgdirname == "":
            imgdirname = "."
        basenames = os.listdir(imgdirname)
        for basename in basenames:
            if basename.lower() == imgbasename.lower():
                imgfilename2 = os.path.join(imgdirname,basename)
                break
        else:
            imgstub = os.path.splitext(imgbasename)[0][:8].lower()
            for basename in basenames:
                stub = os.path.splitext(basename)[0][:8].lower()
                if imgstub == stub:
                    imgfilename2 = os.path.join(imgdirname,basename)
                    break
            else:
                raise
        return open(imgfilename2,"rb"), imgfilename2


def search_for_file_zipfile(zfo,imgfilename):
    try:
        return zfo.open(imgfilename,"r"), imgfilename
    except KeyError:
        filenames = zfo.namelist()
        for filename in filenames:
            if filename.lower() == imgfilename.lower():
                imgfilename2 = filename
                break
        else:
            imgstub = os.path.splitext(imgfilename)[0][:8].lower()
            for filename in filenames:
                stub = os.path.splitext(filename)[0][:8].lower()
                if stub == imgstub:
                    imgfilename2 = filename
                    break
            else:
                raise
        return zfo.open(imgfilename2,"r"), imgfilename2


# Utility for resizing an image

def _next_power_of_2(n):
    assert n > 0
    i = 1
    while i < n:
        i *= 2
    return i


# Check if PIL is available; use it if so.  Otherwise use the GUI's
# image loader

if config.IMAGE_LOAD_PACKAGE in ("PIL_or_pygame","PIL_or_pyglet"):

    try:
        from PIL import Image
    except ImportError:
        config.IMAGE_LOAD_PACKAGE = config.IMAGE_LOAD_PACKAGE.rsplit('_',1)[1]
    else:
        config.IMAGE_LOAD_PACKAGE = "PIL"


# PIL

if config.IMAGE_LOAD_PACKAGE == "PIL":

    from PIL import Image

    def load_image_PIL(flo,hint=None):
        if not hasattr(flo,"seek") or not hasattr(flo,"tell"):
            sflo = io.StringIO(flo.read())
        else:
            sflo = flo
        img = Image.open(sflo)
        if img.mode not in ("RGB","RGBA"):
            img = img.convert("RGB")
        owidth, oheight = img.size
        width = _next_power_of_2(owidth)
        height = _next_power_of_2(oheight)
        if (width,height) != (owidth,oheight):
            img = img.resize((width,height),Image.BICUBIC)
        if img.mode == 'RGB':
            format = "RGBX"
        elif img.mode == 'RGBA':
            format = "RGBA"
        data = img.tostring("raw",format,0,-1)
        return ImageData(data,format,width,height)

    load_image_specific = load_image_PIL


# pyglet, which is currently slow

elif config.IMAGE_LOAD_PACKAGE == "pyglet":

    import pyglet.image
    import pyglet.gl
    import ctypes

    def load_image_pyglet(flo,hint=None):
        pic = pyglet.image.load(hint,file=flo)
        if "A" in pic.format:
            format = "RGBA"
        else:
            format = "RGBX"
        owidth = pic.width
        oheight = pic.height
        data = pic.get_data(format,owidth*len(format))
        width = _next_power_of_2(owidth)
        height = _next_power_of_2(oheight)
        if (width,height) != (owidth,oheight):
            ndata = ctypes.c_buffer(width*height*4)
            pyglet.gl.gluScaleImage(
                pyglet.gl.GL_RGBA,owidth,oheight,pyglet.gl.GL_UNSIGNED_BYTE,
                data,width,height,pyglet.gl.GL_UNSIGNED_BYTE,ndata)
            data = ndata.raw
        return ImageData(data,format,width,height)        

    load_image_specific = load_image_pyglet


# pygame

elif config.IMAGE_LOAD_PACKAGE == "pygame":

    import pygame

    def load_image_pygame(flo,hint=""):
        if not hasattr(flo,"seek"):
            sflo = io.StringIO(flo.read())
        else:
            sflo = flo
        surf = pygame.image.load(sflo,hint)
        if surf.get_flags() & pygame.SRCALPHA:
            format = "RGBA"
        else:
            format = "RGBX"
        owidth,oheight = surf.get_size()
        width = _next_power_of_2(owidth)
        height = _next_power_of_2(oheight)
        if (width,height) != (owidth,oheight):
            if pygame.version.vernum >= (1,8):
                surf = pygame.transform.smoothscale(surf,(width,height))
            else:
                surf = pygame.transform.scale(surf,(width,height))
        data = pygame.image.tostring(surf,format,True)
        return ImageData(data,format,width,height)        

    load_image_specific = load_image_pygame


# Indicates there being no intention to load images

elif config.IMAGE_LOAD_PACKAGE == "none":
    
    pass


# Bad value

else:

    raise ValueError('Dice3DS.example.config.IMAGE_LOAD_PACKAGE '
                     'is set to an invalid value ("%.200r")'
                     % config.IMAGE_LOAD_PACKAGE)
