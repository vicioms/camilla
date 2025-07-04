import numpy as np
import torch

def compute_edge_lengths(verts, tris):
    tri_verts = verts[tris]
    v0,v1,v2 = tri_verts[:,0], tri_verts[:,1], tri_verts[:,2]
    edge_10 = v1 - v0
    edge_20 = v2 - v0
    edge_21 = v2 - v1
    return np.stack([np.linalg.norm(edge_21, axis=0),\
                    np.linalg.norm(edge_20, axis=0),\
                    np.linalg.norm(edge_10, axis=0)], axis=1)
def torch_compute_edge_lengths(verts, tris):
    tri_verts = verts[tris]
    v0,v1,v2 = tri_verts[:,0], tri_verts[:,1], tri_verts[:,2]
    edge_10 = v1 - v0
    edge_20 = v2 - v0
    edge_21 = v2 - v1
    return torch.stack([torch.linalg.norm(edge_21, axis=0),\
                    torch.linalg.norm(edge_20, axis=0),\
                    torch.linalg.norm(edge_10, axis=0)], axis=1)
def compute_triangle_area_vectors(verts, tris):
    return 0.5*np.cross(verts[tris[:, 1],:] - verts[tris[:, 0],:],verts[tris[:, 2],:] - verts[tris[:, 0],:])
def torch_compute_triangle_area_vectors(verts, tris):
    return 0.5*torch.cross(verts[tris[:, 1],:] - verts[tris[:, 0],:],verts[tris[:, 2],:] - verts[tris[:, 0],:])
def compute_triangle_normals(verts, tris, return_areas = False):
    area_vecs = compute_triangle_area_vectors(verts, tris)
    areas = np.linalg.norm(area_vecs, axis=-1)
    if(return_areas):
        return area_vecs/areas[:,None], areas
    else:
        return area_vecs/areas[:,None]
def torch_compute_triangle_normals(verts, tris, return_areas = False):
    area_vecs = torch_compute_triangle_area_vectors(verts, tris)
    areas = torch.linalg.norm(area_vecs, dim=-1)
    if(return_areas):
        return area_vecs/areas[:,None], areas
    else:
        return area_vecs/areas[:,None]
def compute_vertex_normals(verts, tris):
    tri_area_vectors = compute_triangle_area_vectors(verts, tris)
    vert_normal_vectors = np.zeros_like(verts)
    np.add.at(vert_normal_vectors, tris[:,0], tri_area_vectors)
    np.add.at(vert_normal_vectors, tris[:,1], tri_area_vectors)
    np.add.at(vert_normal_vectors, tris[:,2], tri_area_vectors)
    return vert_normal_vectors/np.linalg.norm(vert_normal_vectors, axis=-1, keepdims=True)
def compute_vertex_normals_from_area_vectors(num_vertices, tris, tri_area_vectors):
    vert_normal_vectors = np.zeros((num_vertices,3))
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
def torch_compute_triangle_cotangents(verts, tris, return_areas = False):
    tri_verts = verts[tris]
    v0,v1,v2 = tri_verts[:,0], tri_verts[:,1], tri_verts[:,2]
    edge_10 = v1 - v0
    edge_20 = v2 - v0
    edge_21 = v2 - v1
    areas = torch.linalg.norm(torch.cross(edge_10, edge_20), dim=1)
    cot_0 = torch.sum(edge_10*edge_20, dim=1)/areas
    cot_1 = -torch.sum(edge_10*edge_21, dim=1)/areas
    cot_2 = torch.sum(edge_20*edge_21, dim=1)/areas
    cot = torch.stack([cot_0, cot_1, cot_2], dim=1)
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
