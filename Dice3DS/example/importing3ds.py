# importing3ds.py -*- py-indent-offset: 4 -*-

"""Example of how to import 3DS files into Blender.

Background: The most recent versions of Blender include 3DS
Import/Export plugins, which work ok.  However, like most of the
plugins, they are not so flexible and easy to modify if you need to
change something (say, for example, to implement a custom way to deal
with smoothing).  Dice3DS, however, lets you whip up very concise and
straightforward conversion routines.  So if you need to wheel and deal
with importing 3DS files, I'd suggest starting from here instead.

The drawback of all this is that you'd have to install Python and
numpy, whereas the offical plugin works in Blender without having to
install either.

This module is *not* a Blender plugin.  It's more like a library.  You
could import it from a plugin, or straight from the text area.

"""
from __future__ import division

# Warning: this module is example code but hasn't been updated in
# awhile.  I have come across a few reports that says it doesn't work
# in the recent versions of Blender.  That wouldn't be surprising,
# since the Blender API changes all the time.  However, it still runs
# on my version of Blender (2.49b), so it should be up-to-date enough
# for a quick fix.

from builtins import range
from past.utils import old_div
import os

from Dice3DS import dom3ds
from PIL import Image
import Blender


def getcolor(cc):
    if type(cc) is dom3ds.COLOR_24:
        return old_div(cc.red,255.0), old_div(cc.blue,255.0), old_div(cc.green,255.0)
    if type(cc) is dom3ds.COLOR_F:
        return cc.red, cc.blue, cc.green
    raise ValueError("unknown color chunk %s" % cc.tag)


def getimage(filename):
    try:
        image = Image.open(filename)
    except IOError:
        for dfilename in os.listdir('.'):
            if dfilename.lower() == filename.lower():
                image = Image.open(dfilename)
                break
        else:
            raise
    if image.format in ('PNG','JPEG'):
        return image.filename
    nfilename = os.path.splitext(image.filename)[0] + ".png"
    image.save(nfilename)
    return nfilename


def import3ds(filename):
    """Import a 3DS file into the current scene of Blender.

    Usage:

        import3ds(filename)

    Implementation notes:

    1. Must be run from within Blender, of course.

    2. It does not handle smoothing data at all: output is all solid.

    3. If a texture cannot be found, it does a case-insensitive search
       of the directory; useful on case-sensitive systems when reading
       a 3DS file made on a case-insensitive systems by some ignorant
       fellow.

    4. If a texture is not in PNG or JPEG format, it saves a copy of
       the texture as a PNG file in the same directory.  Blender uses
       the copy for the texture, since Blender only understands JPEG
       and PNG.  Useful for when said ignorant fellow uses a BMP file.

    """

    dom = dom3ds.read_3ds_file(filename,tight=False)

    material_dict = {}
    texture_dict = {}

    b_scene = Blender.Scene.getCurrent()

    for j,d_material in enumerate(dom.mdata.materials):
        b_mat = Blender.Material.New("Material%d" % j)
        b_mat.rgbCol = getcolor(d_material.diffuse.color)
        b_mat.specCol = (0.0,0.0,0.0)
        material_dict[d_material.name.value] = b_mat
        if d_material.texmap is not None:
            b_image = Blender.Image.Load(
                getimage(d_material.texmap.filename.value))
            texture_dict[d_material.name.value] = b_image

    for j,d_nobj in enumerate(dom.mdata.objects):
        if type(d_nobj.obj) != dom3ds.N_TRI_OBJECT:
            continue
        b_obj = Blender.Object.New("Mesh","Object%d" % j)
        b_mesh = Blender.NMesh.New("Mesh%d" % j)
        for d_point in d_nobj.obj.points.array:
            b_vert = Blender.NMesh.Vert(d_point[0],d_point[1],d_point[2])
            b_mesh.verts.append(b_vert)
        d_texverts = [ tuple(tpt) for tpt in d_nobj.obj.texverts.array ]
        for d_face in d_nobj.obj.faces.array:
            vpt = [ b_mesh.verts[int(d_face[i])] for i in range(3) ]
            tpt = [ d_texverts[int(d_face[i])] for i in range(3) ]
            b_face = Blender.NMesh.Face(vpt)
            b_face.uv = tpt
            b_mesh.faces.append(b_face)
        for k,d_material in enumerate(d_nobj.obj.faces.materials):
            b_mesh.materials.append(material_dict[d_material.name])
            for i in d_material.array:
                b_mesh.faces[int(i)].mat = k
                b_image = texture_dict.get(d_material.name)
                if b_image is not None:
                    b_mesh.faces[int(i)].image = b_image
        b_matrix = Blender.Mathutils.Matrix(
            *[tuple(r) for r in d_nobj.obj.matrix.array])
        b_obj.setMatrix(b_matrix)
        b_obj.link(b_mesh)
        b_scene.link(b_obj)
