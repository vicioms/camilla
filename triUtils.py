import numpy as np
import scipy.sparse as sparse
from numba import njit
import networkx as nx
from scipy.spatial import KDTree

def compute_edge_lengths(verts, tris):
    tri_verts = verts[tris]
    v0,v1,v2 = tri_verts[:,0], tri_verts[:,1], tri_verts[:,2]
    edge_10 = v1 - v0
    edge_20 = v2 - v0
    edge_21 = v2 - v1
    return np.stack([np.linalg.norm(edge_21, axis=0),\
                    np.linalg.norm(edge_20, axis=0),\
                    np.linalg.norm(edge_10, axis=0)], axis=1)
@njit
def compute_taubin_matrices(verts, tris, vert_normals, tri_areas):
    M = np.zeros((verts.shape[0], 3, 3))
    W = np.zeros(verts.shape[0])
    for tri_idx,tri in enumerate(tris):
        for a in range(3):
            a_p = (a-1) % 3
            a_n = (a+1) % 3
            normal = vert_normals[tri[a],:]
            for b in [a_p, a_n]:
                edge_b_a = verts[tri[b],:] - verts[tri[a],:]
                tangent_taubin = (np.eye(3) - normal[:,None]*normal[None,:]) @ edge_b_a
                tangent_taubin /= np.linalg.norm(tangent_taubin)
                kappa_taubin = 2*(normal*edge_b_a).sum()/np.sum(edge_b_a**2.0)
                M[tri[a],:,:] += tri_areas[tri_idx]*kappa_taubin*(tangent_taubin[:,None]*tangent_taubin[None,:])
                W[tri[a]] += tri_areas[tri_idx]
    return M/W[:,None,None]
def compute_taubin_principal_curvatures(taubin_matrices, vert_normals, threshold=1e-8):
    put_minus_flag = np.linalg.norm(np.array([1,0,0])[None,:] -  vert_normals, axis=1) >  np.linalg.norm(np.array([1,0,0])[None,:] +  vert_normals, axis=1)
    H_vector = np.zeros_like(vert_normals)
    H_vector[~put_minus_flag, :] = np.array([1,0,0])[None,:] +  vert_normals[~put_minus_flag,:]
    H_vector[put_minus_flag, :] = np.array([1,0,0])[None,:] -  vert_normals[put_minus_flag,:]
    H_vector = H_vector/np.linalg.norm(H_vector, axis=1, keepdims=True)
    Q_matrix = np.eye(3)[None,:,:] - 2*H_vector[:,:,None]*H_vector[:,None,:]
    reduced_matrix =  np.einsum("aji,ajk,akl->ail", Q_matrix, taubin_matrices, Q_matrix)[:,1:,1:]
    pc_vals = np.linalg.eigvalsh(reduced_matrix)
    kappa_1 = 3*pc_vals[:,0] - pc_vals[:,1]
    kappa_2 = 3*pc_vals[:,1] - pc_vals[:,0]
    return kappa_1, kappa_2
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
def compute_vertex_barycentric_areas(verts, tris, tri_areas):
    barycentric_areas = np.zeros(verts.shape[0])
    np.add.at(barycentric_areas, tris[:,0], tri_areas)
    np.add.at(barycentric_areas, tris[:,1], tri_areas)
    np.add.at(barycentric_areas, tris[:,2], tri_areas)
    barycentric_areas /= 3.0
    return barycentric_areas
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
def compute_spatial_graph_clustering(points, values, min_val, max_val, max_distance, num_max_clusters, internal_interval=True):
    tree = KDTree(points)
    sparse_distance_matrix = tree.sparse_distance_matrix(tree, max_distance)
    graph = nx.from_scipy_sparse_array(sparse_distance_matrix)
    if(internal_interval):
        graph.remove_nodes_from(np.argwhere((values <= min_val)+(values >= max_val)).flatten())
    else:
        graph.remove_nodes_from(np.argwhere((values >= min_val)*(values <= max_val)).flatten())
    clusters = [ np.array(list(cs)) for cs in sorted(list(nx.connected_components(graph)), key=len, reverse=True )]
    if(num_max_clusters>0):
        return clusters[:num_max_clusters]
    else:
        return clusters
def compute_edgelist(tris, with_unique_undirected_edges=False):
    ii = tris[:,[0,1,2]]
    jj = tris[:,[1,2,0]]
    edgelist=  np.stack([ii,jj],axis=0).reshape(2, tris.shape[0]*3)
    if(with_unique_undirected_edges):
        np.unique(np.sort(edgelist, axis=0), axis=1)
    else:
        return edgelist
def compute_edgelist_with_opposite_vertex(tris):
    ii = tris[:,[0,1,2]]
    jj = tris[:,[1,2,0]]
    vv = tris[:,[2,0,1]]
    edgelist =  np.stack([ii,jj, vv],axis=0).reshape(3, tris.shape[0]*3)
    return edgelist[[0,1,],...], edgelist[2,...]
def compute_graph_clustering(edgelist, values, min_val, max_val,  internal_interval=True, num_max_clusters=-1, min_num_points=10):
    if(edgelist.shape[0] == 2):
        graph = nx.from_edgelist(edgelist.T)
    else:
        graph = nx.from_edgelist(edgelist)
    if(internal_interval):
        graph.remove_nodes_from(np.argwhere((values <= min_val)+(values >= max_val)).flatten())
    else:
        graph.remove_nodes_from(np.argwhere((values >= min_val)*(values <= max_val)).flatten())
    clusters = [ np.array(list(cs)) for cs in sorted(list(nx.connected_components(graph)), key=len, reverse=True ) if len(cs) >= min_num_points]
    if(num_max_clusters>0):
        return clusters[:num_max_clusters]
    else:
        return clusters