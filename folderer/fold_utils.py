import numpy as np
from math import sqrt
import networkx as nx
from scipy import sparse
from scipy.spatial import ConvexHull

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

        
def plane_mesh_slice(vertices, triangles, plane_origin, plane_normal, epsilon=0):
    # plane/line intersection:
    # n * (p - p0) = 0
    # p = vi + t * (vj - vi)
    # n * (vi + t * (vj - vi) - p0) = 0
    # t = n * (p0 - vi) / n * (vj - vi)
    signed_distances = (vertices - plane_origin) @ plane_normal
    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    s0, s1, s2 = signed_distances[i0], signed_distances[i1], signed_distances[i2]
    def edge_crosses(si,sj):
        return np.logical_and(si > epsilon, sj < -epsilon) | np.logical_and(si < -epsilon, sj > epsilon)
    c01_crosses = edge_crosses(s0, s1)
    c12_crosses = edge_crosses(s1, s2)
    c20_crosses = edge_crosses(s2, s0)
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
def get_longest_shortest_path(graph):
    best_pair = max([ (u, *max(dist.items(), key=lambda x: x[1]))   for u, dist in nx.all_pairs_shortest_path_length(graph)], key=lambda x: x[2])
    return nx.shortest_path(graph, best_pair[0], best_pair[1])

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
