import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fabs

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _tri_crosses(double s_0, double s_1, double s_2, double eps) noexcept nogil:
    """
    Check if a triangle crosses the plane defined by epsilon.
    
    Parameters:
    ---------- 
    s_0, s_1, s_2 : signed distances from the three vertices to the plane.
    eps : tolerance.

    Returns
    ----------
    bint : True if the triangle intersects the plane
    """
    if (s_0 > eps and s_1 > eps and s_2 > eps):
        return False
    if (s_0 < -eps and s_1 < -eps and s_2 < -eps):
        return False
    # coplanar case, skip:
    if (fabs(s_0) < eps and fabs(s_1) < eps and fabs(s_2) < eps):
        return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _edge_crosses(double s_i, double s_j, double eps) noexcept nogil:
    if (s_i > eps and s_j < -eps) or (s_i < -eps and s_j > eps):
        return True
    if fabs(s_i) < eps and fabs(s_j) > eps:
        return True
    if fabs(s_j) < eps and fabs(s_i) > eps:
        return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def slice_mesh(
    double[:,::1] vertices,
    int[:,::1] faces,
    double[:, ::1] plane_normals,
    double[:, ::1] plane_origins,
    double eps):

    cdef Py_ssize_t n_verts  = vertices.shape[0]
    cdef Py_ssize_t n_faces  = faces.shape[0]
    cdef Py_ssize_t n_planes  = plane_normals.shape[0]

    # vertex variables
    cdef Py_ssize_t i
    # face  variables
    cdef Py_ssize_t f
    cdef int idx_0, idx_1, idx_2

    # plane variables
    cdef Py_ssize_t p
    cdef double nx,ny,nz,offset
    cdef double s_0, s_1, s_2



    # precompute signed distances
    cdef double[:, ::1] signed_distances = np.empty((n_verts, n_planes), dtype=np.float64)
    for p in range(n_planes):
        nx = plane_normals[p, 0]
        ny = plane_normals[p, 1]
        nz = plane_normals[p, 2]
        offset = (nx * plane_origins[p, 0] +
            ny * plane_origins[p, 1] +
            nz * plane_origins[p, 2])

        for i in range(n_verts):
            signed_distances[i, p] = (nx * vertices[i, 0] + 
                                    ny * vertices[i, 1] + 
                                    nz * vertices[i, 2] - offset)


    # count of how many times each plane intersects the mesh
    cdef cnp.int64_t[::1] intersection_counts = np.zeros(n_planes, dtype=np.int64)
    for p in range(n_planes):
        for f in range(n_faces):
            s_0 = signed_distances[faces[f, 0], p]
            s_1 = signed_distances[faces[f, 1], p]
            s_2 = signed_distances[faces[f, 2], p]

            if _tri_crosses(s_0, s_1, s_2, eps):
                intersection_counts[p] += 1
    
    # total number of intersections across all planes
    # also we save the starting index of each plane
    # since we are gonna store all intersections in a single array, 
    # we need to know where each plane's intersections start
    cdef cnp.int64_t[::1] start_indices = np.empty(n_planes, dtype=np.int64)
    cdef Py_ssize_t total_int = 0
    for p in range(n_planes):
        start_indices[p] = total_int
        total_int += intersection_counts[p]


    # now we know how many intersections there are, we can allocate the output array
    # the second dimension is 2 because each intersection can be a line segment defined by 2 points
    cdef double[:, :, ::1] segments = np.empty((total_int, 2, 3), dtype=np.float64)
    cdef cnp.int64_t[::1] intersection_tri_indices = np.empty(total_int,  dtype=np.int64)
    cdef int n_hits
    cdef double[6] hits_buf
    cdef double t
    cdef Py_ssize_t j

    for p in range(n_planes):
        j = start_indices[p]
        for f in range(n_faces):
            idx_0 = faces[f, 0]
            idx_1 = faces[f, 1]
            idx_2 = faces[f, 2]
            s_0 = signed_distances[idx_0, p]
            s_1 = signed_distances[idx_1, p]
            s_2 = signed_distances[idx_2, p]

            if not _tri_crosses(s_0, s_1, s_2, eps):
                continue


            n_hits = 0
            if _edge_crosses(s_0, s_1, eps):
                t = s_0 / (s_0 - s_1)
                hits_buf[n_hits*3 + 0] = vertices[idx_0, 0] + t * (vertices[idx_1, 0] - vertices[idx_0, 0])
                hits_buf[n_hits*3 + 1] = vertices[idx_0, 1] + t * (vertices[idx_1, 1] - vertices[idx_0, 1])
                hits_buf[n_hits*3 + 2] = vertices[idx_0, 2] + t * (vertices[idx_1, 2] - vertices[idx_0, 2])
                n_hits += 1

            if  n_hits < 2 and _edge_crosses(s_1, s_2, eps):
                t = s_1 / (s_1 - s_2)
                hits_buf[n_hits*3 + 0] = vertices[idx_1, 0] + t * (vertices[idx_2, 0] - vertices[idx_1, 0])
                hits_buf[n_hits*3 + 1] = vertices[idx_1, 1] + t * (vertices[idx_2, 1] - vertices[idx_1, 1])
                hits_buf[n_hits*3 + 2] = vertices[idx_1, 2] + t * (vertices[idx_2, 2] - vertices[idx_1, 2])
                n_hits += 1

            if n_hits < 2 and _edge_crosses(s_2, s_0, eps):
                t = s_2 / (s_2 - s_0)
                hits_buf[n_hits*3 + 0] = vertices[idx_2, 0] + t * (vertices[idx_0, 0] - vertices[idx_2, 0])
                hits_buf[n_hits*3 + 1] = vertices[idx_2, 1] + t * (vertices[idx_0, 1] - vertices[idx_2, 1])
                hits_buf[n_hits*3 + 2] = vertices[idx_2, 2] + t * (vertices[idx_0, 2] - vertices[idx_2, 2])
                n_hits += 1

            if n_hits == 2:
                segments[j, 0, 0] = hits_buf[0]
                segments[j, 0, 1] = hits_buf[1]
                segments[j, 0, 2] = hits_buf[2]
                segments[j, 1, 0] = hits_buf[3]
                segments[j, 1, 1] = hits_buf[4]
                segments[j, 1, 2] = hits_buf[5]
                intersection_tri_indices[j] = f
                j += 1

    return np.asarray(segments), np.asarray(intersection_tri_indices), np.asarray(intersection_counts)