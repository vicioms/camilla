import numpy as np
from scipy.special import logsumexp

class SinkhornProcrustes:
    @staticmethod
    def _log_sinkhorn_run(loga, logb, f, g, C, eps, num_steps):
        for _ in range(num_steps):
            f = eps * (loga - logsumexp((-C + g[None, :]) / eps, axis=1))
            g = eps * (logb - logsumexp((-C + f[:, None]) / eps, axis=0))
        return f, g

    def __init__(self, p, p0, fit_scale=True):
        p = np.asarray(p, dtype=float)
        p0 = np.asarray(p0, dtype=float)

        self.fit_scale = fit_scale

        self.p = p
        self.p0 = p0

        self.p_mean = p.mean(axis=0)
        self.p0_mean = p0.mean(axis=0)

        self.p_scale = np.sqrt(np.mean(np.sum((p - self.p_mean) ** 2, axis=1))) if fit_scale else 1.0
        self.p0_scale = np.sqrt(np.mean(np.sum((p0 - self.p0_mean) ** 2, axis=1))) if fit_scale else 1.0

        self.x = (p - self.p_mean) / self.p_scale
        self.x0 = (p0 - self.p0_mean) / self.p0_scale
        self.x0_sq = np.sum(self.x0 * self.x0, axis=1)[None, :]

        self.n = len(self.x)
        self.m = len(self.x0)
        self.loga = -np.log(self.n)
        self.logb = -np.log(self.m)

        self.f = np.zeros(self.n)
        self.g = np.zeros(self.m)

        d = p.shape[1]
        self.R = np.eye(d)
        self.s = 1.0

    def _transform_x(self):
        return self.s * (self.x @ self.R.T)

    def _compute_cost_matrix(self):
        x_t = self._transform_x()
        x_t_sq = np.sum(x_t * x_t, axis=1)[:, None]
        C = x_t_sq + self.x0_sq - 2 * (x_t @ self.x0.T)
        return np.maximum(C, 0.0)

    def _log_sinkhorn_step(self, cost_matrix, eps, num_sinkhorn_iterations):
        self.f, self.g = self._log_sinkhorn_run(
            self.loga, self.logb, self.f, self.g, cost_matrix, eps, num_sinkhorn_iterations
        )

    def _weights_from_cost(self, cost_matrix, eps):
        return np.exp((self.f[:, None] + self.g[None, :] - cost_matrix) / eps)

    def _procrustes_step(self, cost_matrix, eps):
        weights = self._weights_from_cost(cost_matrix, eps)

        cross_cov = self.x0.T @ weights.T @ self.x

        U, S, Vt = np.linalg.svd(cross_cov, full_matrices=False)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1.0
            R = U @ Vt

        p_i = np.sum(weights, axis=1)
        den = np.sum(p_i[:, None] * (self.x ** 2))
        if self.fit_scale:
            s = np.sum(S) / den
        else:
            s = 1.0
        self.R = R
        self.s = s

    def run(self, eps_start, eps_end, num_sinkhorn_iterations, num_steps, eps_log_decay=True, return_score=False, return_weight_matrix=False):
        if eps_log_decay:
            eps_values = np.geomspace(eps_start, eps_end, num_steps)
        else:
            eps_values = np.linspace(eps_start, eps_end, num_steps)

        for step, eps in enumerate(eps_values):
            cost_matrix = self._compute_cost_matrix()
            self._log_sinkhorn_step(cost_matrix, eps, num_sinkhorn_iterations)
            self._procrustes_step(cost_matrix, eps)

        if return_score:
            score = np.sum(self._weights_from_cost(cost_matrix, eps_end) * cost_matrix)
            
        s_true = self.s * self.p0_scale / self.p_scale
        t_true = self.p0_mean - s_true * (self.p_mean @ self.R.T)

        return_list = [self.R, s_true, t_true]
        if return_score:
            return_list.append(score)
        if return_weight_matrix:
            return_list.append(self._weights_from_cost(cost_matrix, eps_end))
        return tuple(return_list)
    
    @staticmethod
    def get_transformation_matrix(R, s, t):
        d = R.shape[0]
        M = np.eye(d + 1)
        M[:d, :d] = s * R
        M[:d, d] = t
        return M
    
    @staticmethod
    def get_inv_transformation_matrix(R, s, t):
        d = R.shape[0]
        M_inv = np.eye(d + 1)
        M_inv[:d, :d] = R.T / s
        M_inv[:d, d] = -R.T @ t / s
        return M_inv
    
    @staticmethod
    def transform(points, R, s, t):
        return s * (points @ R.T) + t
    
    @staticmethod
    def inverse_transform(points, R, s, t):
        return (points - t) @ R / s