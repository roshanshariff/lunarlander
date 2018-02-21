# glmodel.py

"""Classes for rendering 3DS models in OpenGL.

Defines some classes (based on Dice3DS.example.basicmodel) with some
additional methods to draw the model in OpenGL, or create a display
list to do so.

"""

from Dice3DS.example import config, basicmodel, gltexture

from Dice3DS.example.anygl import *

import numpy


class GLMaterial(basicmodel.BasicMaterial):
    """Subclass of BasicMaterial that sets OpenGL material properties."""

    def set_material(self,has_texture):
        if self.twosided:
            side = GL_FRONT_AND_BACK
        else:
            side = GL_FRONT
        glMaterialfv(side,GL_AMBIENT,gl_float_array(*self.ambient))
        glMaterialfv(side,GL_DIFFUSE,gl_float_array(*self.diffuse))
        glMaterialfv(side,GL_SPECULAR,gl_float_array(*self.specular))
        glMaterialf(side,GL_SHININESS,self.shininess*128.0)
        if self.texture and has_texture:
            self.texture.enable()
            self.texture.bind()

    def unset_material(self,has_texture):
        if self.texture and has_texture:
            self.texture.disable()


class GLMesh(basicmodel.BasicMesh):
    """Subclass of BasicMesh that renders the mesh in OpenGL."""

    def render_normals(self,length=0.1):
        glPushAttrib(GL_LIGHTING_BIT)
        glDisable(GL_LIGHTING)
        glColor3f(1.0,1.0,0.0)
        n = len(self.norms)
        a = numpy.empty((n*2,3),numpy.float32)
        a[0::2,:] = self.points
        a[1::2,:] = self.points + length*self.norms
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,gl_pointer(a))
        glDrawArrays(GL_LINES,0,n*2)
        glDisableClientState(GL_VERTEX_ARRAY)
        glPopAttrib()

    def render_nomaterials(self):
        glMaterialfv(GL_FRONT,GL_AMBIENT,gl_float_array(0.2,0.2,0.2,1.0))
        glMaterialfv(GL_FRONT,GL_DIFFUSE,gl_float_array(0.8,0.8,0.8,1.0))
        glMaterialfv(GL_FRONT,GL_SPECULAR,gl_float_array(0.0,0.0,0.0,1.0))
        glMaterialf(GL_FRONT,GL_SHININESS,0.0)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3,GL_FLOAT,0,gl_pointer(self.points))
        glNormalPointer(GL_FLOAT,0,gl_pointer(self.norms))
        glDrawArrays(GL_TRIANGLES,0,len(self.points))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

    def render_materials(self):
        has_tverts = self.tverts is not None
        for material,faces in self.matarrays:
            n = len(faces)
            if n == 0:
                continue
            faceelements = numpy.empty(3*n,numpy.int32)
            faceelements[0::3] = numpy.asarray(faces,numpy.int32)*3
            faceelements[1::3] = faceelements[0::3]+1
            faceelements[2::3] = faceelements[1::3]+1
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3,GL_FLOAT,0,gl_pointer(self.points))
            glNormalPointer(GL_FLOAT,0,gl_pointer(self.norms))
            if (not material.texture or not has_tverts
                or not material.texture.real):
                has_texture = False
            else:
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glTexCoordPointer(2,GL_FLOAT,0,gl_pointer(self.tverts))
                has_texture = True
            material.set_material(has_texture)
            glDrawElements(GL_TRIANGLES,n*3,GL_UNSIGNED_INT,
                           gl_pointer(faceelements))
            material.unset_material(has_texture)
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    def render(self):
        if self.matarrays:
            self.render_materials()
        else:
            self.render_nomaterials()


class GLModel(basicmodel.BasicModel):
    """Subclass of BasicModel that renders the model in OpenGL.

    Provides two methods:

        render() - issue the OpenGL commands to draw this model
        create_dl() - create an OpenGL display list of this model,
                and return the handle.

    """

    meshclass = GLMesh
    matclass = GLMaterial

    def render(self):
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_1D)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        for m in self.meshes:
            m.render()
        glPopAttrib()

    def create_dl(self):
        dl = glGenLists(1)
        if dl == 0:
            raise GLError("cannot allocate display list")
        glNewList(dl,GL_COMPILE)
        self.render()
        glEndList()        
        return dl
