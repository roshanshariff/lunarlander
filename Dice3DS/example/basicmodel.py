# basicmodel.py

"""Basic abstract classes representing a 3DS model.

Defines some classes that represent objects and materials of a 3DS
file in a more convienient form. It has methods to convert from the
DOM format. The classes can serve as base classes for more advanced
uses.

"""
from __future__ import division

from builtins import range
from builtins import object
from past.utils import old_div
import math

import numpy

from Dice3DS import dom3ds, util


def _dotchain(first,*rest):
    matrix = first
    for next in rest:
        matrix = numpy.dot(matrix,next)
    return matrix


def _colorf(color,alpha,default):
    if color is None:
        return (default,default,default,alpha)
    if type(color) is dom3ds.COLOR_24:
        return (old_div(color.red,255.0),old_div(color.green,255.0),
                old_div(color.blue,255.0),alpha)
    return (color.red,color.green,color.blue,alpha)


def _pctf(pct,default):
    if pct is None:
        return default
    if type(pct) is dom3ds.INT_PERCENTAGE:
        return old_div(pct.value,100.0)
    return pct.value


class BasicMaterial(object):
    """Represents a material from a 3DS file.

    This class, instances of which a BasicModel instance is
    usually responsible for creating, lists the material
    information.

        mat.name - name of the mateiral
        mat.ambient - ambient color
        mat.diffuse - diffuse color
        mat.specular - specular color
        mat.shininess - shininess exponent
        mat.texture - texture object
        mat.twosided - whether to paint both sides

    """

    def __init__(self,name,ambient,diffuse,specular,
                 shininess,texture,twosided):
        self.name = name
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.texture = texture
        self.twosided = twosided


class BasicMesh(object):
    """Represents a single mesh from a 3DS file.

    This class, instances of which a BasicModel instance is
    usually responsible for creating, takes a mesh data passed in
    from a 3DS DOM, reduces the data somewhat, and calculates
    normals.

        mesh.name - name of the mesh
        mesh.matarrays - list of materials and their corresponding
            faces
        mesh.points - numpy array of points, one row per
            vertex, each set of three adjacent rows representing
            a triangle
        mesh.norms - corresponding array of normal vectors
        mesh.tverts - corresponding array of texture coordinates

    """
    
    def __init__(self,name,facearray,pointarray,smarray,
                 tvertarray,matrix,matarrays,
                 nfunc=util.calculate_normals_by_angle_subtended):

        self.name = name
        self.matarrays = matarrays

        if matrix is None:
            matrix = numpy.identity(4,numpy.float32)
        pointarray = util.translate_points(pointarray,matrix)
        self.points,self.norms = nfunc(pointarray,facearray,smarray)

        if tvertarray is not None:
            self.tverts = tvertarray[facearray.ravel()]
        else:
            self.tverts = None


class BasicModel(object):
    """Represents a basic model from a 3DS file.

    This is an example of a more usable and direct representation
    of a 3DS model.  It can sometimes be hard to operate on data
    while it's sitting in a 3DS DOM; this class and the related
    classes BasicMesh and BasicMaterial make it more accessible.

    """

    meshclass = BasicMesh
    matclass = BasicMaterial

    def __init__(self,dom,load_texture_func,
                 nfunc=util.calculate_normals_by_angle_subtended,
                 frameno=0):
        self.load_texture_func = load_texture_func
        self.frameno = frameno
        try:
            self.extract_materials(dom)
            self.extract_keymatrices(dom)
            self.extract_meshes(dom,nfunc)
        finally:
            del self.load_texture_func

    def extract_materials(self,dom):
        self.materials = {}
        for mat in dom.mdata.materials:
            m = self.create_material(mat)
            self.materials[m.name] = m

    def create_material(self,mat):
        name = mat.name.value
        alpha = 1.0 - _pctf(mat.transparency
                            and mat.transparency.pct,0.0)
        ambient = _colorf(mat.ambient
                          and mat.ambient.color,alpha,0.2)
        diffuse = _colorf(mat.diffuse
                          and mat.diffuse.color,alpha,0.8)
        specular = _colorf(mat.specular
                           and mat.specular.color,alpha,0.0)
        shininess = _pctf(mat.shininess
                          and mat.shininess.pct,0.0)
        texture = (mat.texmap and self.load_texture_func(
            mat.texmap.filename.value))
        twosided = mat.two_side is not None
        return self.matclass(name,ambient,diffuse,specular,
                     shininess,texture,twosided)

    def extract_keymatrices(self,dom):
        self.keymatrix = {}
        if self.frameno < 0:
            return
        if not hasattr(dom,'kfdata') or dom.kfdata is None:
            return

        def getid(index,node):
            if node.node_id is not None:
                return node.node_id.id
            return index

        nodebyid = {}
        keymatrixbyid = {}
        for i,node in enumerate(dom.kfdata.object_nodes):
            nodebyid[getid(i,node)] = node

        def extract_one_keymatrix(node,id):
            matrix = self.create_keymatrix(node)
            parentid = node.node_hdr.parent 
            if parentid != 65535:
                if parentid not in keymatrixbyid:
                    extract_one_keymatrix(nodebyid[parentid],parentid)
                matrix = numpy.dot(matrix,keymatrixbyid[parentid])
            keymatrixbyid[id] = matrix
            self.keymatrix[node.node_hdr.name] = matrix

        for i,node in enumerate(dom.kfdata.object_nodes):
            if node.node_hdr.name in self.keymatrix:
                continue
            extract_one_keymatrix(node,getid(i,node))

    def create_keymatrix(self,kfnode):

        def interpolate_key(keys,attributes):
            for i in range(len(keys)):
                if keys[i].frameno == self.frameno:
                    return [ getattr(keys[i],attr) for attr in attributes ]
                if keys[i].frameno > self.frameno:
                    if i == 0:
                        return [ getattr(keys[0],attr) for attr in attributes ]
                    values = []
                    t = (old_div((self.frameno-keys[i-1].frameno), (keys[i].frameno-keys[i-1].frameno)))
                    for attr in attributes:
                        a = getattr(keys[i-1],attr)
                        b = getattr(keys[i],attr)
                        values.append(a*(1.0-t)+b*t)
                    return values
            return [ getattr(keys[-1],attr) for attr in attributes ]

        pvtmat = numpy.identity(4,numpy.float32)
        if kfnode.pivot is not None:
            pvtmat[0,3] = -kfnode.pivot.pivot_x
            pvtmat[1,3] = -kfnode.pivot.pivot_y
            pvtmat[2,3] = -kfnode.pivot.pivot_z
        posmat = numpy.identity(4,numpy.float32)
        if kfnode.pos_track is not None:
            posx,posy,posz = interpolate_key(
                kfnode.pos_track.keys,["pos_x","pos_y","pos_z"])
            posmat[0,3] = -posx
            posmat[1,3] = -posy
            posmat[2,3] = -posz
        rotmat = numpy.identity(4,numpy.float32)
        if kfnode.rot_track is not None:
            angle,axisx,axisy,axisz = interpolate_key(
                kfnode.rot_track.keys,["angle","axis_x","axis_y","axis_z"])
            if abs(angle) > 0.0001:
                v = numpy.array((axisx,axisy,axisz),numpy.float32)
                u = old_div(v, math.sqrt(numpy.dot(v,v)))
                s = numpy.array(((0,-u[2],u[1]),
                                 (u[2],0,-u[0]),
                                 (-u[1],u[0],0)),numpy.float32)
                p = numpy.outer(u,u)
                i = numpy.identity(3,numpy.float32)
                m = p + math.cos(angle)*(i-p) + math.sin(angle)*s
                rotmat[0:3,0:3] = m
        sclmat = numpy.identity(4,numpy.float32)
        if kfnode.scl_track is not None:
            sclx,scly,sclz = interpolate_key(
                kfnode.scl_track.keys,["scl_x","scl_y","scl_z"])
            sclmat[0,0] = old_div(1.0, sclx)
            sclmat[1,1] = old_div(1.0, scly)
            sclmat[2,2] = old_div(1.0, sclz)
        return _dotchain(pvtmat,sclmat,rotmat,posmat)

    def extract_meshes(self,dom,nfunc):
        self.meshes = []
        kfobj = {}
        for nobj in dom.mdata.objects:
            obj = nobj.obj
            if type(obj) is not dom3ds.N_TRI_OBJECT:
                continue
            if obj.faces is None:
                continue
            if obj.faces.nfaces < 1:
                continue
            kfnode = kfobj.get(nobj.name)
            mesh = self.create_mesh(nobj.name,obj,nfunc)
            self.meshes.append(mesh)

    def create_mesh(self,name,nto,nfunc):
        facearray = numpy.array(nto.faces.array[:,:3])
        smarray = nto.faces.smoothing and nto.faces.smoothing.array
        pointarray = nto.points.array
        tvertarray = nto.texverts and nto.texverts.array
        matrix = nto.matrix and nto.matrix.array
        keymatrix = self.keymatrix.get(name)
        if keymatrix is not None:
            matrix = numpy.dot(matrix,keymatrix)
        matarrays = []
        for m in nto.faces.materials:
            matarrays.append((self.materials[m.name],m.array))
        return self.meshclass(name,facearray,pointarray,smarray,
                              tvertarray,matrix,matarrays,nfunc)

    def center_of_gravity(self):
        cg = numpy.zeros((3,),numpy.float32)
        n = 0
        for mesh in self.meshes:
            cg += sum(mesh.points)
            n += len(mesh.points)
        return old_div(cg,n)

    def bounding_box(self):
        ul = numpy.maximum.reduce([ numpy.maximum.reduce(mesh.points)
                                    for mesh in self.meshes ])
        ll = numpy.minimum.reduce([ numpy.minimum.reduce(mesh.points)
                                    for mesh in self.meshes ])
        return ul,ll
