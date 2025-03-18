

# Import necessary Cython and NumPy headers
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from libcpp.vector cimport vector
from libcpp.map import map
import numpy as np
cimport numpy as cnp

cdef class edgelist:
    cdef vector[double] lengths
    cdef vector[int] edges  #constructed from v_to + V*v_from
    cdef map[int,int] twins
    cdef int num_vertices
    cdef int num_triangles

    def build_edgelist(self, cnp.ndarray[double, ndim=2] verts, cnp.ndarray[int, ndim=2] tris):
        self.num_vertices = verts.shape[0]
        self.num_triangles = tris.shape[0]
        for tri_idx in range(0, self.num_triangles):
            tri = tris[tri_idx,:]
            edge_10 = verts[tri[1],:] - verts[tri[0],:]
            edge_21 = verts[tri[2],:] - verts[tri[1],:]
            edge_02 = verts[tri[0],:] - verts[tri[2],:]
            len_10 = sqrt(edge_10[0]*edge_10[0]+edge_10[1]*edge_10[1]+edge_10[2]*edge_10[2])
            len_21 = sqrt(edge_21[0]*edge_21[0]+edge_21[1]*edge_21[1]+edge_21[2]*edge_21[2])
            len_02 = sqrt(edge_02[0]*edge_02[0]+edge_02[1]*edge_02[1]+edge_02[2]*edge_02[2])

            s = (len_10+len_21+len_02)/2
            area = sqrt(s*(s-len_10)*(s-len_21)*(s-len_02))

            self.lengths.push_back(len_10)
            self.edges.push_back(tri[1]+self.num_vertices*tri[0])

            self.lengths.push_back(len_21)
            self.edges.push_back(tri[2]+self.num_vertices*tri[1])

            self.lengths.push_back(len_02)
            self.edges.push_back(tri[0]+self.num_vertices*tri[2])
        return


    







