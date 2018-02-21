# modelloader.py

"""Example of loading 3DS models.

Provides functions to load a 3DS model and creating a GLModel (or
BasicModel) from it. Shows how to load models from the filesystem, or
directly from a zip file.

"""

# Warning: this module is example code but hasn't been checked in
# awhile.  I don't see any bugs but it's untested.

from future import standard_library
standard_library.install_aliases()
import io
import zipfile

from Dice3DS.example import gltexture, glmodel


def load_model_from_filesystem(filename,modelclass=glmodel.GLModel,
                               texture_options=()):

    """Load a model from the filesystem.

        model = load_model_from_filesystem(filename,
                    modelclass=glmodel.GLModel,
                    texture_options=())

    This loads a model, where the textures are to be found in the
    filesystem.  filename is the 3DS file to load.  Any textures
    listed in it are expected to be found relative to the current
    directory.

    It creates a model of the given class, and creates textures
    with the given texture options.

    """

    texcache = {}
    def load_texture(texfilename):
        if texfilename in texcache:
            return texcache[texfilename]
        tex = gltexture.Texture(texfilename,*texture_options)
        texcache[texfilename] = tex
        return tex
    
    dom = dom3ds.read_3ds_file(filename)
    return modelclass(dom,load_texture)


def load_model_from_zipfile(zipname,arcname,texture_options=()):
    """Load a model from a zipfile.

        model = load_model_from_filesystem(zipfilename,
                    archivename, modelclass=glmodel.GLModel,
                    texture_options=())

        This loads a model, where the 3DS file and the textures are
        found inside a zipfile.  zipfilename is the name of the
        zipfile.  arcfilename is the name of the 3DS file inside the
        zipfile archive.  Any textures listed in it are expected to be
        found inside the zipfile as well.

    It creates a model of the given class, and creates textures
        with the given texture options.

    """

    texcache = {}
    def load_texture(texarchivename):
        if texarchivename in texcache:
            return texcache[texarchivename]
        tex = gltexture.Texture((zfo,texarchivename),*texture_options)
        texcache[texarchivename] = tex
        return tex
    
    zfo = ZipFile(zipname,"r")
    try:
        dom = dom3ds.read_3ds_mem(zfo.read(arcname))
        return modelclass(dom,load_texture)
    finally:
        zfo.close()

