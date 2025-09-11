import numpy as np
import pymeshfix
import open3d as o3d
from typing import Union
def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    ret_mesh = mesh.remove_duplicated_triangles()
    ret_mesh = ret_mesh.remove_duplicated_vertices()
    ret_mesh = ret_mesh.remove_degenerate_triangles()
    ret_mesh = ret_mesh.remove_non_manifold_edges()
    ret_mesh = ret_mesh.remove_unreferenced_vertices()

    # Keep only the largest connected component
    clusters, lengths, _ = ret_mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    lengths = np.asarray(lengths)
    largest_cluster = np.argmax(lengths)
    ret_mesh.remove_triangles_by_index(
        np.where(clusters != largest_cluster)[0]
    )
    ret_mesh = ret_mesh.remove_unreferenced_vertices()

    # Remove non-manifold vertices
    nm_verts = ret_mesh.get_non_manifold_vertices()
    if len(nm_verts) > 0:
        ret_mesh.remove_vertices_by_index(nm_verts)

    # Final clean-up
    ret_mesh = ret_mesh.remove_non_manifold_edges()
    ret_mesh = ret_mesh.remove_unreferenced_vertices()

    return ret_mesh
def load_and_process_mesh(file_path, scale : Union[np.ndarray,float,None], flip0 : bool = False, flip1 : bool = False, flip2 : bool = False, return_as_open3d_mesh = False):
    """
    Load a mesh from a file using Open3D, repair it using pymeshfix, and return the repaired mesh.

    Parameters:
    - file_path: str, path to the mesh file.
    
    Returns:
    - vertices: np.ndarray, array of vertex coordinates.
    - triangles: np.ndarray, array of face indices.
    """
    # Load the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Convert Open3D mesh to numpy arrays
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if scale is not None:
        if np.isscalar(scale):
            vertices = vertices * scale
        else:   
            assert len(scale) == 3
            vertices = vertices * scale.reshape(1,3)
    
    # Use pymeshfix to repair the mesh
    #meshfix = pymeshfix.MeshFix(vertices, triangles)
    #meshfix.repair()
    #
    ## Get the repaired vertices and faces
    #vertices = meshfix.points
    #triangles = meshfix.faces
    
    # Create a new Open3D mesh with the repaired data
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh = clean_mesh(mesh)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if flip0:
        vertices[:, 0] = vertices[:, 0].max()+vertices[:, 0].min()-vertices[:, 0]
    if flip1:
        vertices[:, 1] = vertices[:, 1].max()+vertices[:, 1].min()-vertices[:, 1]
    if flip2:
        vertices[:, 2] = vertices[:, 2].max()+vertices[:, 2].min()-vertices[:, 2]
    if(return_as_open3d_mesh):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        return mesh
    else:       
        return vertices, triangles
def estimate_mean_curvature_scale(mean_curvatures : np.ndarray, use_positive = True, use_bootstrap = True, num_bootstraps = 10000, frac_samples_per_bootstrap = 0.1):
    curvatures_to_use = mean_curvatures[mean_curvatures>0] if use_positive else mean_curvatures[mean_curvatures<0]
    if use_bootstrap:
        indices_to_use = np.random.choice(np.arange(len(curvatures_to_use)), size = (num_bootstraps, int(len(curvatures_to_use)*frac_samples_per_bootstrap)), replace = True)
        curvature_scales = np.mean(curvatures_to_use[indices_to_use], axis=1)
        return np.mean(curvature_scales), np.std(curvature_scales)
    else:
        return np.mean(curvatures_to_use), np.std(curvatures_to_use)
