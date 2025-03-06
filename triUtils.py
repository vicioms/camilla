import numpy as np
import scipy.sparse as sparse
def compute_triangle_area_vectors(verts, tris):
    """ Return the area vector associated to each triangle.

    Args:
        verts (_type_): (V,3) array
        tris (_type_): (F,3) array

    Returns:
        _type_: Area vectors (F,3)
    """
    return 0.5*np.cross(verts[tris[:, 1],:] - verts[tris[:, 0],:],verts[tris[:, 2],:] - verts[tris[:, 0],:])
def compute_triangle_normals(verts, tris, return_areas = False):
    """_summary_

    Args:
        verts (_type_): _description_
        tris (_type_): _description_
        return_areas (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    area_vecs = compute_triangle_area_vectors(verts, tris)
    areas = np.linalg.norm(area_vecs, axis=-1)
    if(return_areas):
        return area_vecs/areas[:,None], areas
    else:
        return area_vecs/areas[:,None]
def compute_vertex_normals(verts, tris):
    """_summary_

    Args:
        verts (_type_): _description_
        tris (_type_): _description_

    Returns:
        _type_: _description_
    """
    tri_area_vectors = compute_triangle_area_vectors(verts, tris)
    vert_normal_vectors = np.zeros_like(verts)
    np.add.at(vert_normal_vectors, tris[:,0], tri_area_vectors)
    np.add.at(vert_normal_vectors, tris[:,1], tri_area_vectors)
    np.add.at(vert_normal_vectors, tris[:,2], tri_area_vectors)
    return vert_normal_vectors/np.linalg.norm(vert_normal_vectors, axis=-1, keepdims=True)
def compute_vertex_normals_from_area_vectors(num_vertices, tris, tri_area_vectors):
    """_summary_

    Args:
        verts (_type_): _description_
        tris (_type_): _description_

    Returns:
        _type_: _description_
    """
    vert_normal_vectors = np.zeros_like((num_vertices,3))
    np.add.at(vert_normal_vectors, tris[:,0], tri_area_vectors)
    np.add.at(vert_normal_vectors, tris[:,1], tri_area_vectors)
    np.add.at(vert_normal_vectors, tris[:,2], tri_area_vectors)
    return vert_normal_vectors/np.linalg.norm(vert_normal_vectors, axis=-1, keepdims=True)
def compute_triangle_cotangents(verts, tris, return_areas = False):
    tri_verts = verts[tris]
    v0,v1,v2 = tri_verts[:,0], tri_verts[:,1], tri_verts[:,2]
    edge_10 = v1 - v0
    edge_20 = v2 - v0
    edge_21 = v2 - v1
    areas = np.linalg.norm(np.cross(edge_10, edge_20), axis=1)
    cot_0 = np.sum(edge_10*edge_20, axis=1)/areas
    cot_1 = -np.sum(edge_10*edge_21, axis=1)/areas
    cot_2 = np.sum(edge_20*edge_21, axis=1)/areas
    cot = np.stack([cot_0, cot_1, cot_2], axis=1)
    if(return_areas):
        return cot, areas
    else:
        return cot
def compute_triangle_angles(verts, tris):
    tri_verts = verts[tris]
    v0,v1,v2 = tri_verts[:,0], tri_verts[:,1], tri_verts[:,2]
    edge_10 = v1 - v0
    edge_20 = v2 - v0
    edge_21 = v2 - v1
    areas = np.linalg.norm(np.cross(edge_10, edge_20), axis=1)
    theta_0 = np.arctan2(areas, np.sum(edge_10*edge_20, axis=1))
    theta_1 = np.arctan2(areas, -np.sum(edge_10*edge_21, axis=1))
    theta_2 = np.arctan2(areas, np.sum(edge_20*edge_21, axis=1))
    return np.stack([theta_0, theta_1, theta_2], axis=1)
def compute_cot_matrix(verts, tris, return_areas = False):
    if(return_areas):
        cot, tri_areas = compute_triangle_cotangents(verts, tris, return_areas=True)
    else:
        cot = compute_triangle_cotangents(verts, tris, return_areas=False)
    ii = tris[:, [1,2,0]]
    jj = tris[:, [2,0,1]]
    idx = np.stack([ii,jj], axis=0).reshape(2, tris.shape[0]*3)
    cot_matrix = sparse.coo_array((cot.flatten(), idx))
    cot_matrix += cot_matrix.transpose()
    if(return_areas):
        return cot_matrix, tri_areas
    else:
        return cot_matrix
def compute_cot_laplacian(verts, tris, normalize_by_areas = True, return_areas = False):
    cot_matrix, tri_areas = compute_cot_matrix(verts, tris, return_areas=True)
    barycentric_areas = np.zeros(verts.shape[0])
    np.add.at(barycentric_areas, tris[:,0], tri_areas)
    np.add.at(barycentric_areas, tris[:,1], tri_areas)
    np.add.at(barycentric_areas, tris[:,2], tri_areas)
    barycentric_areas /= 3.0
    lapl_matrix = sparse.dia_array((cot_matrix.sum(axis=1),0), shape=cot_matrix.shape).tocoo() - cot_matrix
    if(normalize_by_areas):
        lapl_matrix =  lapl_matrix @ sparse.dia_array((1.0/barycentric_areas,0), shape=cot_matrix.shape).tocoo()
    if(return_areas):
        return lapl_matrix, barycentric_areas
    else:
        return lapl_matrix
    
