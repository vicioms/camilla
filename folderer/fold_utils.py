import numpy as np
from math import sqrt
import networkx as nx
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.interpolate import interp1d, CubicSpline
from scipy.interpolate import splprep, splev, make_splprep
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import depth_first_order

import igl
from typing import List, Optional, Tuple, Dict
from numpy.typing import NDArray
from scipy.spatial import KDTree

from multiprocessing import Pool
from functools import partial

import plotly.graph_objects as go
import plotly.express as px

from tqdm import tqdm

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
def fit_conic(x,y):
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    U, S, V = np.linalg.svd(D)
    P = V[-1, :]
    return P
def fit_ellipsoid(x, y, z):
    """
    Fit ellipsoid:
        X^T A X + 2 b^T X + c = 0

    Returns params:
        [Axx, Ayy, Azz, Axy, Axz, Ayz, bx, by, bz, c]

    where A =
        [[Axx, Axy, Axz],
         [Axy, Ayy, Ayz],
         [Axz, Ayz, Azz]]

    and equation is:
        X^T A X + 2 b^T X + c = 0
    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    if x.size < 9 or y.size != x.size or z.size != x.size:
        return np.full(10, np.nan)

    P = np.column_stack([x, y, z])

    # normalize
    mu = P.mean(axis=0)
    s = np.sqrt(np.mean(np.sum((P - mu)**2, axis=1)))
    if s == 0 or not np.isfinite(s):
        return np.full(10, np.nan)

    X = (P - mu) / s
    xn, yn, zn = X[:, 0], X[:, 1], X[:, 2]

    # design matrix for:
    # x^2, y^2, z^2, 2xy, 2xz, 2yz, 2x, 2y, 2z, 1
    D = np.column_stack([
        xn**2,
        yn**2,
        zn**2,
        2*xn*yn,
        2*xn*zn,
        2*yn*zn,
        2*xn,
        2*yn,
        2*zn,
        np.ones_like(xn),
    ])

    try:
        _, _, vh = np.linalg.svd(D, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full(10, np.nan)

    p = vh[-1]

    A = np.array([
        [p[0], p[3], p[4]],
        [p[3], p[1], p[5]],
        [p[4], p[5], p[2]],
    ])

    b = np.array([p[6], p[7], p[8]])
    c = p[9]

    # enforce positive definite A by flipping sign if needed
    evals = np.linalg.eigvalsh(A)
    if np.all(evals < 0):
        A = -A
        b = -b
        c = -c
        evals = -evals

    if not np.all(evals > 0):
        return np.full(10, np.nan)

    # center in normalized coordinates
    try:
        center_n = -np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(10, np.nan)

    # translated equation:
    # (x-center)^T A (x-center) = k
    k = b @ np.linalg.solve(A, b) - c

    if k <= 0 or not np.isfinite(k):
        return np.full(10, np.nan)

    # normalize so RHS = 1
    A = A / k
    b = b / k
    c = c / k

    # denormalize homogeneous quadric
    # normalized coord: xn = (x - mu) / s
    T = np.eye(4)
    T[:3, :3] = np.eye(3) / s
    T[:3, 3] = -mu / s

    Qn = np.zeros((4, 4))
    Qn[:3, :3] = A
    Qn[:3, 3] = b
    Qn[3, :3] = b
    Qn[3, 3] = c

    Q = T.T @ Qn @ T

    A0 = Q[:3, :3]
    b0 = Q[:3, 3]
    c0 = Q[3, 3]

    return np.array([
        A0[0, 0],
        A0[1, 1],
        A0[2, 2],
        A0[0, 1],
        A0[0, 2],
        A0[1, 2],
        b0[0],
        b0[1],
        b0[2],
        c0,
    ])
def get_ellipsoid_geometry(p):
    A = np.array([
        [p[0], p[3], p[4]],
        [p[3], p[1], p[5]],
        [p[4], p[5], p[2]],
    ])
    b = np.array([p[6], p[7], p[8]])
    c = p[9]

    center = -np.linalg.solve(A, b)
    k = b @ np.linalg.solve(A, b) - c

    S = A / k
    evals, evecs = np.linalg.eigh(S)

    axes = 1.0 / np.sqrt(evals)

    return center, axes, evecs
def get_triangle_adjacency_matrix(triangles):
    F = triangles.shape[0]

    # --- build undirected edges ---
    edges = np.stack([
        triangles[:, [0,1]],
        triangles[:, [1,2]],
        triangles[:, [2,0]],
    ], axis=1)                      # (F,3,2)

    edges = np.sort(edges, axis=2)  # canonical ordering
    edges = edges.reshape(-1, 2)    # (3F,2)

    tri_ids = np.repeat(np.arange(F), 3)

    # --- sort edges ---
    order = np.lexsort((edges[:,1], edges[:,0]))
    edges_s = edges[order]
    tris_s  = tri_ids[order]

    # --- find shared edges ---
    same = np.all(edges_s[1:] == edges_s[:-1], axis=1)
    idx = np.where(same)[0]

    t0 = tris_s[idx]
    t1 = tris_s[idx + 1]

    # --- build symmetric adjacency matrix ---
    rows = np.concatenate([t0, t1])
    cols = np.concatenate([t1, t0])

    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(F, F)).tocsr()

#def _edge_crosses(si,sj, epsilon):
#        return np.logical_and(si > epsilon, sj < -epsilon) | np.logical_and(si < -epsilon, sj > epsilon)
def _edge_crosses(si, sj, epsilon=0):
    # treat zero as positive (consistent tiebreak)
    si = np.where(np.abs(si) <= epsilon, epsilon, si)
    sj = np.where(np.abs(sj) <= epsilon, epsilon, sj)
    return (si > 0) != (sj > 0)
def plane_mesh_slice(vertices, triangles, plane_origin, plane_normal, epsilon=0):
    # plane/line intersection:
    # n * (p - p0) = 0
    # p = vi + t * (vj - vi)
    # n * (vi + t * (vj - vi) - p0) = 0
    # t = n * (p0 - vi) / n * (vj - vi)
    signed_distances = (vertices - plane_origin) @ plane_normal
    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    s0, s1, s2 = signed_distances[i0], signed_distances[i1], signed_distances[i2]
    
    c01_crosses = _edge_crosses(s0, s1, epsilon)
    c12_crosses = _edge_crosses(s1, s2, epsilon)
    c20_crosses = _edge_crosses(s2, s0, epsilon)
    den01 = (s0-s1)
    den12 = (s1-s2)
    den20 = (s2-s0)

    t01 = np.full( s0.shape, np.nan)
    t12 = np.full( s0.shape, np.nan)
    t20 = np.full( s0.shape, np.nan)

    t01[c01_crosses] = s0[c01_crosses] / den01[c01_crosses]
    t12[c12_crosses] = s1[c12_crosses] / den12[c12_crosses]
    t20[c20_crosses] = s2[c20_crosses] / den20[c20_crosses]

    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
    p01 = v0 + (v1 - v0) * t01[:, None]
    p12 = v1 + (v2 - v1) * t12[:, None]
    p20 = v2 + (v0 - v2) * t20[:, None]
    P = np.stack([p01,p12,p20], axis=1)
    M = np.stack([c01_crosses, c12_crosses, c20_crosses], axis=1)
    E = np.stack([triangles[:,[0,1]], triangles[:,[1,2]], triangles[:,[2,0]]], axis=1)
    good_triangles = np.sum(M, axis=1) == 2
    P_good = P[good_triangles]
    M_good = M[good_triangles]
    E_good = E[good_triangles]
    segments = P_good[M_good].reshape(-1,2,3)

    return segments, np.argwhere(good_triangles).flatten(), E_good[M_good].reshape(-1,2)
def plane_mesh_slices_single_normal(vertices, triangles, plane_origins, plane_normal, epsilon=0):
    v_dot_n = vertices @ plane_normal                     # (num_vertices,)
    o_dot_n = plane_origins @ plane_normal               # (num_planes,)

    # (num_planes, num_vertices)
    signed_distances = v_dot_n[None, :] - o_dot_n[:, None]

    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    # (num_planes, num_triangles)
    s0 = signed_distances[:, i0]
    s1 = signed_distances[:, i1]
    s2 = signed_distances[:, i2]

    c01_crosses = _edge_crosses(s0, s1, epsilon)
    c12_crosses = _edge_crosses(s1, s2, epsilon)
    c20_crosses = _edge_crosses(s2, s0, epsilon)

    den01 = s0 - s1
    den12 = s1 - s2
    den20 = s2 - s0

    t01 = np.full(s0.shape, np.nan, dtype=vertices.dtype)
    t12 = np.full(s0.shape, np.nan, dtype=vertices.dtype)
    t20 = np.full(s0.shape, np.nan, dtype=vertices.dtype)

    t01[c01_crosses] = s0[c01_crosses] / den01[c01_crosses]
    t12[c12_crosses] = s1[c12_crosses] / den12[c12_crosses]
    t20[c20_crosses] = s2[c20_crosses] / den20[c20_crosses]

    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]   # (num_triangles, 3)

    # (num_planes, num_triangles, 3)
    p01 = v0[None, :, :] + (v1 - v0)[None, :, :] * t01[:, :, None]
    p12 = v1[None, :, :] + (v2 - v1)[None, :, :] * t12[:, :, None]
    p20 = v2[None, :, :] + (v0 - v2)[None, :, :] * t20[:, :, None]

    # candidate points and masks
    # P: (num_planes, num_triangles, 3_edges, 3_xyz)
    # M: (num_planes, num_triangles, 3_edges)
    P = np.stack([p01, p12, p20], axis=2)
    M = np.stack([c01_crosses, c12_crosses, c20_crosses], axis=2)

    # edge vertex ids: (num_triangles, 3_edges, 2)
    E = np.stack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=1
    )

    # triangles cut in exactly two edges
    good = np.sum(M, axis=2) == 2                        # (num_planes, num_triangles)

    # indices of (plane, triangle) that produce one segment
    plane_ids, tri_ids = np.nonzero(good)

    # select only good plane-triangle pairs
    P_good = P[good]                                     # (num_good, 3_edges, 3_xyz)
    M_good = M[good]                                     # (num_good, 3_edges)
    E_good = E[tri_ids]                                  # (num_good, 3_edges, 2)

    # pick the 2 valid points / edges for each good pair
    segments = P_good[M_good].reshape(-1, 2, 3)          # (num_good, 2, 3)
    crossed_edges = E_good[M_good].reshape(-1, 2, 2)     # (num_good, 2, 2)

    return segments, plane_ids, tri_ids, crossed_edges
def get_longest_shortest_path(graph):
    best_pair = max([ (u, *max(dist.items(), key=lambda x: x[1]))   for u, dist in nx.all_pairs_shortest_path_length(graph)], key=lambda x: x[2])
    return nx.shortest_path(graph, best_pair[0], best_pair[1])
def get_approx_longest_shortest_path(graph):
    # BFS from arbitrary node → find far endpoint u
    start = next(iter(graph))
    lengths = nx.single_source_shortest_path_length(graph, start)
    u = max(lengths, key=lengths.get)
    
    # BFS from u → find far endpoint v (true diameter on trees, good approx on general graphs)
    lengths_from_u = nx.single_source_shortest_path_length(graph, u)
    v = max(lengths_from_u, key=lengths_from_u.get)
    
    return nx.shortest_path(graph, u, v)
def min_area_enclosing_triangle_from_scipy_hull(hull: ConvexHull, *, return_area=True):
    """
    Minimum-area enclosing triangle for a 2D scipy.spatial.ConvexHull.

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull
        Must be a 2D hull (hull.points.shape[1] == 2).
    return_area : bool
        If True returns (tri, area). Else returns tri only.

    Returns
    -------
    tri : (3,2) float64
        Triangle vertices (x,y) in arbitrary order (OpenCV output).
    area : float
        Triangle area (only if return_area=True).

    Notes
    -----
    Requires OpenCV: pip install opencv-python
    OpenCV expects a *contour*; we feed it the hull polygon vertices in CCW order.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV is required for min-area enclosing triangle. "
            "Install with: pip install opencv-python"
        ) from e

    pts = np.asarray(hull.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected 2D hull.points, got shape {pts.shape}")

    # SciPy hull.vertices gives indices of hull vertices in CCW order (for 2D hulls).
    poly = pts[np.asarray(hull.vertices, dtype=int)]

    # OpenCV contour format: (N,1,2) float32
    contour = poly.astype(np.float32).reshape(-1, 1, 2)

    area, tri = cv2.minEnclosingTriangle(contour)
    tri = np.asarray(tri, dtype=np.float64).reshape(3, 2)
    area = float(area)

    return (tri, area) if return_area else tri
#def get_segment_path_from_triangle_path(tri_adj_graph, tri_indices, tri_to_seg_index):
#    sub = tri_adj_graph.subgraph(tri_indices)
#    tris = list(tri_indices)
#    local_id = {tri: k for k, tri in enumerate(tris)}
#
#    rows, cols = zip(*[(local_id[u], local_id[v]) for u, v in sub.edges()]) if sub.edges() else ([], [])
#    rows = list(rows) + list(cols)
#    cols = list(cols) + list(rows[:len(rows)//2 if rows else 0])
#    # simpler: just use nx for the DFS since connected components already uses it
#    # and the bottleneck is plane_mesh_slice, not this
#    deg = dict(sub.degree())
#    start = next((t for t in tris if deg[t] == 1), tris[0])
#    path = list(nx.dfs_preorder_nodes(sub, start))
#    seg_indices = np.array([tri_to_seg_index[t] for t in path])
#    return path, seg_indices

def get_segment_path_from_triangle_path(tri_adj_graph, tri_indices, tri_to_seg_index):
    sub = tri_adj_graph.subgraph(tri_indices)
    deg = dict(sub.degree())
    tris = list(tri_indices)

    # start from a degree-1 node (endpoint) if it exists, else any node
    start = next((t for t in tris if deg[t] == 1), tris[0])

    # greedy walk: always go to the unvisited neighbor
    path = [start]
    visited = {start}
    current = start
    while True:
        neighbors = [nb for nb in sub.neighbors(current) if nb not in visited]
        if not neighbors:
            break
        current = neighbors[0]  # only one unvisited neighbor if it's a path graph
        visited.add(current)
        path.append(current)

    seg_indices = np.array([tri_to_seg_index[t] for t in path])
    return path, seg_indices

class FoldAnnotation:
    @staticmethod
    def _align_paths_with_dtw(paths):
        
        # Choose a reference path (first path in this case)
        ref_path = paths[0]

        # List to store aligned paths
        aligned_paths = []

        for path in paths:
            # Compute the DTW distance and the alignment path
            distance, alignment_path = fastdtw(ref_path, path, dist=euclidean)
            # Align path to the reference path
            aligned_path = np.array([path[idx] for idx in list(zip(*alignment_path))[1]])
            aligned_paths.append(aligned_path)
        return aligned_paths
    @staticmethod
    def _resample_path(path, num_points):
        
        original_indices = np.linspace(0, 1, len(path))
        target_indices = np.linspace(0, 1, num_points)

        interpolator = interp1d(original_indices, path, axis=0, kind='cubic', fill_value="extrapolate")
        resampled_path = interpolator(target_indices)
        return resampled_path
    @staticmethod
    def _average_aligned_paths(aligned_paths, num_points):
        resampled_paths = [FoldAnnotation._resample_path(path, num_points) for path in aligned_paths]
        stacked_paths = np.stack(resampled_paths)
        avg_path = np.mean(stacked_paths, axis=0)
        err_path = np.sqrt(np.var(stacked_paths, axis=0).sum(axis=-1))
        return avg_path, err_path
    
    @staticmethod
    def _merge_path_djistrka(paths, path_weights, graph, edge_weight_attr='weight'):
        all_nodes = np.concatenate(paths)
        unique_nodes, unique_nodes_counts = np.unique(all_nodes, return_counts=True)
        node_to_index= {node: idx for idx, node in enumerate(unique_nodes)}
        max_count = unique_nodes_counts.max()
        subgraph = nx.subgraph(graph, unique_nodes)

        for u, v, d in subgraph.edges(data=True):
            vote_weight = (unique_nodes_counts[node_to_index[u]] + unique_nodes_counts[node_to_index[v]])/(2*max_count)
            d['combined_' + edge_weight_attr] = d[edge_weight_attr] / vote_weight
        best_path_index = np.argmin(path_weights)
        best_path = paths[best_path_index]
        src, dst = best_path[0], best_path[-1]
        merged_path = nx.dijkstra_path(subgraph, src, dst, weight='combined_' + edge_weight_attr)
        return np.array(merged_path)
    
    
    def __init__(self,
                 vertices, 
                 triangles, 
                 vertex_mean_curvature, 
                 indices,
                 path_through_positive_curvature : bool,
                 alpha  : float,
                 vertex_adj_list = Optional[List[List[int]]],
                 path_quantile_level= 0.1,
                 restrict_path_to_boundary = False,
                 path_weight_quantile_level = 0.9,
                 path_inside_segmentation = True,
                 use_djistrka_merge = True
                 ):
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_mean_curvature = vertex_mean_curvature


        if path_through_positive_curvature:
            pos_quantiles_H = np.quantile(vertex_mean_curvature[vertex_mean_curvature > 0], [0.25, 0.75])
            if(1/alpha < pos_quantiles_H[0] or 1/alpha > pos_quantiles_H[1]):
                print(f"Warning: alpha^-1={1/alpha} is outside the interquartile range for (positive) mean curvature [{pos_quantiles_H[0]:.4f}, {pos_quantiles_H[1]:.4f}]. Might lead to unexpected results. Consider changing alpha.")
            
        else:
            neg_quantiles_H = np.quantile(np.abs(vertex_mean_curvature[vertex_mean_curvature < 0]), [0.25,0.75])
            if(1/alpha < neg_quantiles_H[0] or 1/alpha > neg_quantiles_H[1]):
                print(f"Warning: alpha^-1={1/alpha} is outside the interquartile range for (negative) mean curvature [{neg_quantiles_H[0]:.4f}, {neg_quantiles_H[1]:.4f}]. Might lead to unexpected results. Consider changing alpha.")

        self.indices = indices    
        if vertex_adj_list is not None:
            self.vertex_adj_list = vertex_adj_list
        else:
            self.vertex_adj_list = igl.adjacency_list(triangles)
        self.vertex_adj_graph = nx.from_dict_of_lists({i: nbrs for i, nbrs in enumerate(self.vertex_adj_list)})
        self.tri_adj_graph = nx.from_scipy_sparse_array(get_triangle_adjacency_matrix(triangles))
        self.vertex_normals = igl.per_vertex_normals(self.vertices, self.triangles)
        self.triangle_centers = np.mean(self.vertices[self.triangles], axis=1)
        self.triangle_normals = igl.per_face_normals(self.vertices, self.triangles)
        self.boundary_region_indices = np.unique(np.array(list(nx.edge_boundary(self.vertex_adj_graph, self.indices)))[:,0])
        self._vertex_weights = None

        self._start_indices = None
        self._end_indices = None
        self._vpath_indices = None
        self._vpath_indices_weights = None
        
        self._vpaths = None


        self._tpath_indices = None
        self._tpath_indices_projections = None

        self._merged_vpath_indices = None
        self._merged_vpath = None
        self._merged_vpath_error = None



        self.path_through_positive_curvature = path_through_positive_curvature
        self.alpha = alpha
        self.path_quantile_level = path_quantile_level
        self.restrict_path_to_boundary = restrict_path_to_boundary
        self.path_weight_quantile_level = path_weight_quantile_level
        self.path_inside_segmentation = path_inside_segmentation
        self.use_djistrka_merge = use_djistrka_merge

    def compute_vertex_weights(self):
        if self._vertex_weights is not None:
            return self._vertex_weights
        from_nodes = np.repeat(np.arange(0, len(self.vertices)), [len(nbrs) for nbrs in self.vertex_adj_list])
        to_nodes = np.concatenate(self.vertex_adj_list)
        self._vertex_weights =  np.exp( ( -1.0 if self.path_through_positive_curvature else 1.0) * self.alpha * (self.vertex_mean_curvature[from_nodes] + self.vertex_mean_curvature[to_nodes])/2.0)
        self._vertex_weights[~np.isfinite(self._vertex_weights)] = np.inf
        self.vertex_adj_graph.add_weighted_edges_from(zip(from_nodes, to_nodes, self._vertex_weights))
        
        return self._vertex_weights

    def estimate_typical_edge_length(self, reduce_func = np.median):
        edge_lengths = np.linalg.norm(self.vertices[self.triangles[:, 0]] - self.vertices[self.triangles[:, 1]], axis=-1)
        edge_lengths = np.concatenate([edge_lengths, np.linalg.norm(self.vertices[self.triangles[:, 1]] - self.vertices[self.triangles[:, 2]], axis=-1)])
        edge_lengths = np.concatenate([edge_lengths, np.linalg.norm(self.vertices[self.triangles[:, 2]] - self.vertices[self.triangles[:, 0]], axis=-1)])
        return reduce_func(edge_lengths)

    def estimate_ellipticity(self):
        '''
            We estimate the ellipticity of the region defined by 'indices'. We employ a nematic decomposition of the covariance matrix of the vertices defined by 'indices'. 
            We return two nematic order parameters, S and T, where s is the the nematic order and t is a measure of the biaxiality of the region.
            When S > 0, the region is elongated along the direction of largest variance, and when S < 0, the region is compressed along the direction of smallest variance.
    '''
        points = self.vertices[self.indices]
        points_cov = np.cov(points, rowvar=False, bias=True)
        vals, vecs = np.linalg.eigh(points_cov)

        q_vals = 0.5*np.log(vals)
        q_vals -= q_vals.sum()/3
        if abs(q_vals[0]) < q_vals[2]:
            s_val = q_vals[2]
            t_val = (q_vals[1] - q_vals[0])/2
        else:
            s_val = q_vals[0]
            t_val = (q_vals[2] - q_vals[1])/2

        return s_val, t_val
    def compute_vpath_boundary_indices(self) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
        '''
            Estimates using PCA on vertices selected by 'indices' the extremal vertices, taken along the direction of largest variance. 
            We retain points in the lower and upper quantiles along this direction, as determined by 'path_quantile_level' i.e. 
            quantiles at levels self.path_quantile_level and 1.0 - self.path_quantile_level.
        '''
        if self._start_indices is not None and self._end_indices is not None:
            return self._start_indices, self._end_indices
        points = self.vertices[self.indices]
        points_cov = np.cov(points, rowvar=False, bias=True)
        vals, vecs = np.linalg.eigh(points_cov)
        largest_eigvec = vecs[:, -1]
        projected_points = (points - points.mean(axis=0, keepdims=True)) @ largest_eigvec
        quantiles = np.quantile(projected_points, [self.path_quantile_level, 1.0 - self.path_quantile_level])
        start_indices =  self.indices[(projected_points < quantiles[0])]
        end_indices = self.indices[(projected_points > quantiles[1])]
        if self.restrict_path_to_boundary and len(self.boundary_region_indices) > 0:
            start_indices = np.intersect1d(start_indices, self.boundary_region_indices)
            end_indices = np.intersect1d(end_indices, self.boundary_region_indices)
            if len(start_indices) == 0 or len(end_indices) == 0:
                raise ValueError("No start or end indices found on boundary region. Consider disabling restrict_path_to_boundary.")
        self._start_indices = np.unique(start_indices)
        self._end_indices = np.unique(end_indices)
        return self._start_indices, self._end_indices   
    def compute_vpath_indices(self) -> Tuple[List[NDArray[np.int_]], NDArray[np.float64]]:
        '''
            We find using Djikstra's algorithm the shortest paths between all pairs of start and end indices, as determined by 'compute_vpath_boundary_indices'.
            We employ as weights between vertex 'i' and 'j' the value exp( alpha * (curvature(i) + curvature(j))/2.0) if 'path_through_positive_curvature' is True, 
            and exp( - alpha * (curvature(i) + curvature(j))/2.0) otherwise, where curvature(i) is the mean curvature at vertex i.
        '''
        if self._vpath_indices is not None:
            return self._vpath_indices, self._vpath_indices_weights
        _ = self.compute_vertex_weights()
        subgraph = self.vertex_adj_graph.subgraph(self.indices)
        start_indices, end_indices = self.compute_vpath_boundary_indices()
        paths = []
        path_weights = []
        for v1 in start_indices:
            distances_1, paths_1 = nx.algorithms.single_source_dijkstra(subgraph, v1, weight='weight')
            for v2 in end_indices:
                if(v2  in paths_1):
                    path = paths_1[v2]
                    dist = distances_1[v2]
                    if not np.isfinite(dist):
                        continue
                    paths.append(np.array(path))
                    path_weights.append(dist)
        self._vpath_indices = paths
        self._vpath_indices_weights = np.array(path_weights)
        if self.path_weight_quantile_level is not None and self.path_weight_quantile_level < 1.0:
            weight_threshold = np.quantile(self._vpath_indices_weights, self.path_weight_quantile_level)
            valid_paths_mask = self._vpath_indices_weights <= weight_threshold
            self._vpath_indices = [path for i, path in enumerate(self._vpath_indices) if valid_paths_mask[i]]
            self._vpath_indices_weights = self._vpath_indices_weights[valid_paths_mask]
        return self._vpath_indices, self._vpath_indices_weights
    def compute_vpaths(self) -> List[NDArray[np.float64]]:
        '''
            For each path, we return the 3D coordinates of the vertices along the path. 
        '''
        if self._vpaths is not None:
            return self._vpaths
        paths, _ = self.compute_vpath_indices()
        path_vertices = [self.vertices[path] for path in paths]
        self._vpaths = path_vertices
        return self._vpaths
    
    def merge_vpaths(self, num_points : Optional[int] = None) -> Tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
        '''
            We merge the paths found by 'compute_vpath_indices' into a single path, by first aligning them using Dynamic Time Warping (DTW) and then averaging the aligned paths. 
            We return the indices of the vertices along the merged path, the 3D coordinates of the vertices along the merged path and the error estimate for each point of the merged path.
        '''

        if self.use_djistrka_merge:
            if self._merged_vpath_indices is not None and self._merged_vpath is not None and self._merged_vpath_error is not None:
                return self._merged_vpath_indices, self._merged_vpath, self._merged_vpath_error
            path_indices, path_weights = self.compute_vpath_indices()
            merged_path_indices = self._merge_path_djistrka(path_indices, path_weights, self.vertex_adj_graph, edge_weight_attr='weight')
            merged_path = self.vertices[merged_path_indices]
            self._merged_vpath_indices = merged_path_indices
            self._merged_vpath = merged_path
            self._merged_vpath_error = np.zeros(len(merged_path))
        else:
            if num_points is None:
                estimated_spacing = 2*self.estimate_typical_edge_length()
                num_points = self.estimate_num_points_for_merged_vpaths(estimated_spacing)
            if self._merged_vpath is not None and self._merged_vpath_error is not None:
                if len(self._merged_vpath) == num_points:
                    return self._merged_vpath_indices, self._merged_vpath, self._merged_vpath_error

            aligned_paths = self._align_paths_with_dtw(self.compute_vpaths())
            merged_path, merged_path_error = self._average_aligned_paths(aligned_paths, num_points=num_points)
            if self.path_inside_segmentation:
                _, sub_indices = KDTree(self.vertices[self.indices]).query(merged_path)
            else:
                _, sub_indices = KDTree(self.vertices).query(merged_path)
            self._merged_vpath_indices = self.indices[sub_indices] if self.path_inside_segmentation else sub_indices
            self._merged_vpath = self.vertices[self._merged_vpath_indices]
            self._merged_vpath_error = merged_path_error
        return self._merged_vpath_indices, self._merged_vpath, self._merged_vpath_error

    def compute_tpath_indices(self) -> Tuple[List[NDArray[np.int_]], List[NDArray[np.float64]]]:
        '''
            For each vertex in each path, we find the triangle it belongs to whose plane is closest to the vertex, and return the indices of these triangles, as well as the projection of the vertex onto the plane of the triangle. 
        '''
        if self._tpath_indices is not None and self._paths_on_triangle_indices_projections is not None:
            return self._tpath_indices, self._paths_on_triangle_indices_projections
        paths, _ = self.compute_vpath_indices()
        self._tpath_indices = []
        self._paths_on_triangle_indices_projections = []
        for path in paths:
            candidate_triangles = np.argwhere(np.isin(self.triangles, path).any(axis=1)).flatten()
            
            projection = np.sum(self.triangle_normals[None,candidate_triangles,:] * (self.vertices[path][:,None,:] - self.triangle_centers[None, candidate_triangles, :]), axis=-1)
            correct_indices = np.argmin(np.abs(projection), axis=1)
            self._tpath_indices.append(candidate_triangles[correct_indices])
            self._paths_on_triangle_indices_projections.append(projection[np.arange(len(path)), correct_indices])
        return self._tpath_indices, self._paths_on_triangle_indices_projections 

    def visualize(self, additional_curves : Optional[List] = None,
                  fig : Optional[go.Figure] = None, 
                  mesh_color = 'lightgrey', 
                  display_merged_path = False,
                  display_vpaths = False,
                  fraction_of_vpaths_to_display : float = 1.0,
                  show_fig = True):
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
            title="",
            width=800, height=680,
            scene=dict(
                xaxis_title='x', yaxis_title='y', zaxis_title='z',
                aspectmode='data',
                uirevision="keep"  # preserve camera/zoom
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(itemsizing='constant')
            )
        mesh_trace = go.Mesh3d(
            x=self.vertices[:, 0],
            y=self.vertices[:, 1], 
            z=self.vertices[:, 2],
            i=self.triangles[:, 0],
            j=self.triangles[:, 1],
            k=self.triangles[:, 2],
            color=mesh_color,
            opacity=0.5,
            name='mesh'
        )
        palette = px.colors.qualitative.Plotly
        fig.add_trace(mesh_trace)

        if display_vpaths:
            paths, _ = self.compute_vpath_indices()
            paths_to_display = np.arange(len(paths)) if fraction_of_vpaths_to_display >= 1.0 else np.random.choice(len(paths), int(len(paths) * fraction_of_vpaths_to_display), replace=False)
            for cnt, i in enumerate(paths_to_display):
                fig.add_trace(go.Scatter3d(
                    x=self.vertices[paths[i], 0],
                    y=self.vertices[paths[i], 1],
                    z=self.vertices[paths[i], 2],
                    mode='lines',
                    opacity=0.8,
                    line=dict(color=palette[cnt % len(palette)], width=5),
                ))
        if display_merged_path:
            merged_path_indices, merged_path, merged_path_error = self.merge_vpaths(num_points=100)
            fig.add_trace(go.Scatter3d(
                x=merged_path[:, 0],
                y=merged_path[:, 1],
                z=merged_path[:, 2],
                mode='lines',
                line=dict(color='black', width=8),
                name='merged path'
            )) 

        if additional_curves is not None:
            if isinstance(additional_curves, list):
                for curve in additional_curves:
                    fig.add_trace(go.Scatter3d(
                        x=curve[:, 0],
                        y=curve[:, 1],
                        z=curve[:, 2],
                        mode='lines',
                        line=dict(color='red', width=5),
                        name='additional curve'
                    ))
        if show_fig:
            fig.show()    
        return fig
    
    def estimate_num_points_for_merged_vpaths(self, target_spacing):
        all_num_points = []
        for path in self.compute_vpaths():
            path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=-1))
            num_points = int(np.ceil(path_length / target_spacing))
            all_num_points.append(num_points)
        all_num_points = np.array(all_num_points)
        return int(np.ceil(np.median(all_num_points)))
    
    def construct_fold_cross_sections_via_spline(
        self,
        spline_smoothing_factor: float,
        verbose: bool = False,
        plane_mesh_slice_epsilon : float = 0.0,
        num_points : Optional[int] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[int, List[NDArray[np.float64]]]]:

        merged_path_indices, merged_path, merged_path_error = self.merge_vpaths(num_points=num_points)
        path_spline, u = make_splprep(merged_path.T, s=spline_smoothing_factor)
        curve = np.ascontiguousarray(np.stack(path_spline(u, nu=0)).T, dtype=np.float64)
        curve_der = np.ascontiguousarray(np.stack(path_spline(u, nu=1)).T, dtype=np.float64)
        curve_der2 = np.ascontiguousarray(np.stack(path_spline(u, nu=2)).T, dtype=np.float64)
        curve_tangents = curve_der / np.linalg.norm(curve_der, axis=-1, keepdims=True)
        # Frenet normal
        curve_normals = curve_der2 - (curve_der2 * curve_tangents).sum(-1, keepdims=True) * curve_tangents  # remove tangent component
        curve_normals /= np.linalg.norm(curve_normals, axis=-1, keepdims=True)
        cross_sections = {}
        cross_sections_normals = {}

        #for i in tqdm(range(len(curve)), desc="Slicing mesh with planes", disable=not verbose):
        #    p0 = curve[i]
        #    n = curve_tangents[i]
        #    segments, segment_tri_indices, segment_edges = plane_mesh_slice(
        #        self.vertices, self.triangles, plane_origin=p0, plane_normal=n, epsilon=plane_mesh_slice_epsilon
        #    )
        #    tri_index_to_segment_index = {tri_idx: seg_idx for seg_idx, tri_idx in enumerate(segment_tri_indices)}
        #    segments_tri_subgraph = self.tri_adj_graph.subgraph(segment_tri_indices).copy()
        #    non_adjacent_segments_groups = list(nx.connected_components(segments_tri_subgraph))
#
        #    cross_sections[i] = []
        #    cross_sections_normals[i] = []
        #    for segments_group in non_adjacent_segments_groups:
        #        longest_shortest_path = get_approx_longest_shortest_path(segments_tri_subgraph.subgraph(segments_group))
        #        #longest_shortest_path = get_longest_shortest_path(segments_tri_subgraph.subgraph(segments_group))
        #        longest_shortest_path_normals = self.triangle_normals[longest_shortest_path]
        #        longest_shortest_path_seg_indices = np.array([tri_index_to_segment_index[tri_idx] for tri_idx in longest_shortest_path])
        #        ordered_points = (segments[longest_shortest_path_seg_indices, 1, :] + segments[longest_shortest_path_seg_indices, 0, :]) / 2
        #        cross_sections[i].append(ordered_points)
        #        cross_sections_normals[i].append(longest_shortest_path_normals)


        

        for i in tqdm(range(len(curve)), desc="Slicing mesh with planes", disable=not verbose):
            p0 = curve[i]
            n = curve_tangents[i]
            segments, segment_tri_indices, segment_edges = plane_mesh_slice(
                self.vertices, self.triangles, plane_origin=p0, plane_normal=n, epsilon=plane_mesh_slice_epsilon
            )

            seg_index_of = {tri: i for i, tri in enumerate(segment_tri_indices)}
            segments_tri_subgraph = self.tri_adj_graph.subgraph(segment_tri_indices).copy()
            non_adjacent_segments_groups = list(nx.connected_components(segments_tri_subgraph))

            cross_sections[i] = []
            cross_sections_normals[i] = []
            for group in non_adjacent_segments_groups:
                tri_path, seg_indices = get_segment_path_from_triangle_path(segments_tri_subgraph, group, seg_index_of)
                if tri_path is None:
                    continue
                ordered_points = (segments[seg_indices, 0] + segments[seg_indices, 1]) / 2
                cross_sections[i].append(ordered_points)

                cross_sections_tri_normals = self.triangle_normals[seg_indices]
                # project normals to plane orthogonal to curve tangent
                cross_sections_tri_normals -= (cross_sections_tri_normals * n).sum(axis=-1, keepdims=True) * n
                cross_sections_tri_normals /= np.linalg.norm(cross_sections_tri_normals, axis=-1, keepdims=True)

                cross_sections_normals[i].append(cross_sections_tri_normals)

        return curve, curve_tangents, curve_normals, cross_sections, cross_sections_normals

    @staticmethod
    def estimate_alphas(mean_curvature, use_positive_curvature = True):
        if use_positive_curvature:
            curvature_values = mean_curvature[mean_curvature > 0]
        else:
            curvature_values = -mean_curvature[mean_curvature < 0]
        curvature_values = curvature_values[np.isfinite(curvature_values) & (curvature_values > 0)]
        if len(curvature_values) == 0:
            return 1.0, 1.0
        log_curvatures = np.log(curvature_values)
        alpha_std = 1.0/np.std(log_curvatures)
        alpha_med= np.exp(np.median(1.0/log_curvatures))
        return alpha_std, alpha_med
        
    @staticmethod
    def get_disk_harmonic_coords(vertices, triangles, boundary_indices : Optional[NDArray[np.int_]] = None):
        '''
            We compute the disk harmonic coordinates of the vertices defined by 'boundary_indices', on the mesh defined by 'vertices' and 'triangles'. 
            We return a matrix of size (len(vertices), len(boundary_indices)), where the entry (i,j) is the disk harmonic coordinate of vertex i with respect to boundary vertex j. 
        '''
        if boundary_indices is None:
            boundary_indices = igl.boundary_loop(triangles)
        #boundary_vertices = vertices[boundary_indices]
        #boundary_distances =  np.linalg.norm(np.roll(boundary_vertices, -1, axis=0) - boundary_vertices, axis=1)
        #arc_length_parameter = np.concatenate([[0.0], np.cumsum(boundary_distances)])[:-1]
        #arc_length_parameter /= arc_length_parameter[-1]                                          # normalize to [0,1)
        #theta = 2.0*np.pi * arc_length_parameter
        #boundary_coords = np.c_[np.cos(theta), np.sin(theta)]
        return igl.harmonic(vertices, triangles, boundary_indices, igl.map_vertices_to_circle(vertices, boundary_indices), 1)
    



from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

def arclength_resample(points, N, tolerance=0.0):
    # compute cumulative arc length
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    mask = seg > tolerance
    seg = seg[mask]
    
    s = np.concatenate([[0], np.cumsum(seg)])
    s /= s[-1]  # normalize to [0,1]

    # cubic spline interpolation
    cs = CubicSpline(s, points[np.concatenate([[True], mask])])
    s_uniform = np.linspace(0, 1, N)
    return cs(s_uniform), s_uniform

def optimal_rotation_dst(coeffs, p0, pL, use_smart_init=True):
    a, b = coeffs[:, 0], coeffs[:, 1]
    dl = pL - p0
    k = np.arange(1, len(a) + 1) * np.pi

    def margin(theta):
        a_rot = a * np.cos(theta) - b * np.sin(theta)
        dl_x  = dl[0] * np.cos(theta) - dl[1] * np.sin(theta)
        return (dl_x - (np.abs(a_rot) * k).sum())/np.linalg.norm(dl_x)
    
    thetas = np.linspace(-np.pi/2, np.pi/2, 360)
    if use_smart_init:

        # grid search first, then refine
        
        margins = [margin(t) for t in thetas]
        theta0 = thetas[np.argmax(margins)]

        res = minimize_scalar(lambda t: -margin(t),
                              bounds=(theta0 - np.pi/180*5, theta0 + np.pi/180*5),
                              method='bounded')
        
    else:
        
        res = minimize_scalar(lambda t: -margin(t),
                              bounds=(thetas[0], thetas[-1]),
                              method='bounded')
    return res.x, margin(res.x)
def rotate_dst(coeffs, p0, pL, theta, center=None):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [ np.sin(theta),  np.cos(theta)]])
    
    if center is None:
        center = (p0 + pL) / 2
    
    coeffs_rot = (R @ coeffs.T).T
    p0_rot = R @ (p0 - center) + center
    pL_rot = R @ (pL - center) + center
    
    return coeffs_rot, p0_rot, pL_rot

def curvature_dst(coeffs, p0, pL, s):
    a, b = coeffs[:, 0], coeffs[:, 1]
    k = np.arange(1, len(a) + 1) * np.pi
    ks = np.outer(k, s)

    cos_ks = np.cos(ks)
    sin_ks = np.sin(ks)

    dx  = (pL[0] - p0[0]) + (a[:, None] * k[:, None] * cos_ks).sum(0)
    ddx = -(a[:, None] * k[:, None]**2 * sin_ks).sum(0)
    dy  = (pL[1] - p0[1]) + (b[:, None] * k[:, None] * cos_ks).sum(0)
    ddy = -(b[:, None] * k[:, None]**2 * sin_ks).sum(0)

    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return kappa