import numpy as np
#import pymeshfix
import open3d as o3d
from typing import Union
from math import sqrt
from scipy.spatial import ConvexHull
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
    if flip0:
        vertices[:, 0] = vertices[:, 0].max()+vertices[:, 0].min()-vertices[:, 0]
    if flip1:
        vertices[:, 1] = vertices[:, 1].max()+vertices[:, 1].min()-vertices[:, 1]
    if flip2:
        vertices[:, 2] = vertices[:, 2].max()+vertices[:, 2].min()-vertices[:, 2]

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
    
    if(return_as_open3d_mesh):
        return mesh
    else:       
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
def estimate_mean_curvature_scale(mean_curvatures : np.ndarray, use_positive = True, use_bootstrap = True, num_bootstraps = 10000, frac_samples_per_bootstrap = 0.1):
    curvatures_to_use = mean_curvatures[mean_curvatures>0] if use_positive else mean_curvatures[mean_curvatures<0]
    if use_bootstrap:
        indices_to_use = np.random.choice(np.arange(len(curvatures_to_use)), size = (num_bootstraps, int(len(curvatures_to_use)*frac_samples_per_bootstrap)), replace = True)
        curvature_scales = np.mean(curvatures_to_use[indices_to_use], axis=1)
        return np.mean(curvature_scales), np.std(curvature_scales)
    else:
        return np.mean(curvatures_to_use), np.std(curvatures_to_use)
def orient_centroid(vertices,triangles, vertex_normals):
    should_flip = np.mean((vertices - vertices.mean(axis=0, keepdims=True))*vertex_normals > 0) < 0.5
    if should_flip:
        triangles = triangles[:, [2,1,0]]
        vertex_normals = -vertex_normals
    return triangles, vertex_normals
def orient_centroid_convex_hull(vertices,triangles, vertex_normals):
    hull = ConvexHull(vertices)
    hull_vertices = vertices[hull.vertices]
    hull_center = hull_vertices.mean(axis=0, keepdims=True)
    hull_triangles = triangles[np.isin(triangles, hull.vertices).all(axis=1)]
    hull_triangle_centers = vertices[hull_triangles].mean(axis=1, keepdims=False)
    hull_triangle_normals = vertex_normals[hull_triangles].mean(axis=1, keepdims=False)
    should_flip = np.mean((hull_triangle_centers - hull_center)*hull_triangle_normals > 0) < 0.5
    if should_flip:
        triangles = triangles[:, [2,1,0]]
        vertex_normals = -vertex_normals
    return triangles, vertex_normals
def _fit_ellipse_direct_least_squares(x : np.ndarray, y: np.ndarray) -> np.ndarray:
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    C_matr = np.array([[0, 0, 2],
                  [0,-1, 0],
                  [2, 0, 0]], dtype=float)

    try:
        S3_inv_S2T = np.linalg.solve(S3, S2.T) # this is the negative of the T matrix in Halir and Flusser
    except np.linalg.LinAlgError:
        return np.full(6, np.nan)

    M = S1 - S2 @ S3_inv_S2T

    try:
        C_matr_inv = np.linalg.inv(C_matr)
    except np.linalg.LinAlgError:
        return np.full(6, np.nan)

    evals, evecs = np.linalg.eig(C_matr_inv @ M)

    # sanity checks for real solutions
    real_mask = np.isfinite(evals.real) & (np.abs(evals.imag) < 1e-8)
    evals = evals.real[real_mask]
    evecs = evecs[:, real_mask].real
    if evecs.size == 0:
        return np.full(6, np.nan)

    # ellipse constraint: 4ac - b^2 > 0
    a_, b_, c_ = evecs[0, :], evecs[1, :], evecs[2, :]
    ok = 4*a_*c_ - b_**2
    valid = np.where(ok > 0)[0]
    if valid.size == 0:
        return np.full(6, np.nan)

    # choose the valid eigenvector with smallest |λ| (best behaved)
    k = valid[np.argmin(np.abs(evals[valid]))]
    a1 = evecs[:, k]

    # recover linear terms
    a2 = - S3_inv_S2T @ a1
    return np.hstack([a1, a2])
def fit_ellipse(x : np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit an ellipse to a set of points (x,y) using the Direct Least Squares method by Halir and Flusser (1998).
    https://autotrace.sourceforge.net/WSCG98.pdf
    The ellipse is represented in the conic form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.

    Parameters
    ---------- 
    x : np.ndarray
          x-coordinates of the points 
    y : np.ndarray
          y-coordinates of the points

    return_cond_number : bool, optional
        If True, the function also returns the condition number of the design matrix.
        Default is False.
        
    Returns
    ----------
    p : np.ndarray
        Parameters of the ellipse in the conic form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
        The output is a vector [A,B,C,D,E,F]
        If no ellipse could be fitted, an array filled with np.nan is returned.
    cond : float, optional
        The condition number of the design matrix, only if return_cond_number is True.
        If no ellipse could be fitted, np.nan is returned.
    """
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if x.size < 5 or y.size != x.size:
        return np.full(6, np.nan)
    

    # --- normalization (translation + isotropic scale) ---
    mx, my = x.mean(), y.mean()
    s = np.sqrt(((x - mx)**2 + (y - my)**2).mean())
    if s == 0 or not np.isfinite(s):
        return np.full(6, np.nan)
    xn = (x - mx) / s
    yn = (y - my) / s

    pn = _fit_ellipse_direct_least_squares(xn, yn)
    if np.any(np.isnan(pn)):
        return np.full(6, np.nan)


    # --- denormalize back to original coords ---
    a,b,c,d,e,f = pn
    # x'=(x-mx)/s, y'=(y-my)/s
    A = a / s**2
    B = b / s**2
    C = c / s**2
    D = - (2*a*mx + b*my)/s**2 + d/s
    E = - (b*mx + 2*c*my)/s**2 + e/s
    F = (a*mx**2 + b*mx*my + c*my**2)/s**2 - d*mx/s - e*my/s + f
    p = np.array([A, B, C, D, E, F], dtype=float)

    return p
def get_ellipse_canonical_parameters(p : np.ndarray):
    '''
    Convert conic parameters of an ellipse to canonical parameters.
    Parameters
    ---------- 
    p : np.ndarray
        Parameters of the ellipse in the conic form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
        The input is a vector [A,B,C,D,E,F] or [A,B,C,D,E] (with F=0).
    Returns
    ----------
    params : dict
        A dictionary with the canonical parameters of the ellipse:
        - 'a': semi-major axis length
        - 'b': semi-minor axis length
        - 'x0': x-coordinate of the center
        - 'y0': y-coordinate of the center
        - 'theta': rotation angle of the ellipse (in radians).
    '''

    if(len(p) == 6):
        A,B,C,D,E,F = p
    elif(len(p) == 5):
        A,B,C,D,E = p
        F = 0
    else:
        raise ValueError("Conic parameters must be a vector of length 5 or 6.")
    det = B*B - 4*A*C
    if(det >= 0):
        raise ValueError("The conic is not an ellipse (det >= 0).")
    x0 = (2*C*D - B*E)/det
    y0 = (2*A*E - B*D)/det
    theta = 0.5 * np.arctan2(-B, C-A)
    temp = 2*(A*E*E + C*D*D - B*D*E + det*F)
    a = - sqrt(temp*(A+C + sqrt((A-C)**2 + B*B)))/det
    b = - sqrt(temp*(A+C - sqrt((A-C)**2 + B*B)))/det
    if(a < b):
        a, b = b, a
        theta += np.pi/2
    return {'a': a, 'b': b, 'x0': x0, 'y0': y0, 'theta': theta}
