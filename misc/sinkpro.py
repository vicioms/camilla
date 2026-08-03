import numpy as np
from scipy.special import logsumexp
from tqdm.auto import tqdm
class SinkhornProcrustes:
    """
    Fit a similarity transform

        p0 ~= s * (p @ R.T) + t

    using alternating soft matching and Procrustes updates.

    coupling:
        "sinkhorn":
            doubly stochastic soft matching with uniform marginals.

        "attention":
            one-sided soft assignment.

            If attention_direction == "p_to_p0":
                W[i, j] = probability that p_i matches p0_j.
                Rows sum to 1 / n.

            If attention_direction == "p0_to_p":
                W[i, j] = probability that p0_j matches p_i.
                Columns sum to 1 / m.

    Notes
    -----
    The internal normalized-space transform is

        x0 ~= s * (x @ R.T) + t_norm

    and the returned real-space transform is

        p0 ~= s_true * (p @ R.T) + t_true
    """

    @staticmethod
    def _log_sinkhorn_run(loga, logb, f, g, C, eps, num_steps):
        for _ in range(num_steps):
            f = eps * (loga - logsumexp((-C + g[None, :]) / eps, axis=1))
            g = eps * (logb - logsumexp((-C + f[:, None]) / eps, axis=0))
        return f, g

    @staticmethod
    def _check_points(p, p0):
        p = np.asarray(p, dtype=float)
        p0 = np.asarray(p0, dtype=float)

        if p.ndim != 2 or p0.ndim != 2:
            raise ValueError("p and p0 must be 2D arrays of shape (n_points, dim).")

        if p.shape[1] != p0.shape[1]:
            raise ValueError("p and p0 must have the same ambient dimension.")

        if len(p) == 0 or len(p0) == 0:
            raise ValueError("p and p0 must be non-empty.")

        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(p0)):
            raise ValueError("p and p0 must contain only finite values.")

        return p, p0

    def __init__(
        self,
        p,
        p0,
        fit_scale=True,
        coupling="sinkhorn",
        attention_direction="p_to_p0",
        allow_reflection=False,
        min_scale=0.0,
    ):
        p, p0 = self._check_points(p, p0)

        if coupling not in {"sinkhorn", "attention"}:
            raise ValueError("coupling must be either 'sinkhorn' or 'attention'.")

        if attention_direction not in {"p_to_p0", "p0_to_p"}:
            raise ValueError("attention_direction must be either 'p_to_p0' or 'p0_to_p'.")

        self.fit_scale = fit_scale
        self.coupling = coupling
        self.attention_direction = attention_direction
        self.allow_reflection = allow_reflection
        self.min_scale = min_scale

        self.p = p
        self.p0 = p0

        self.p_mean = p.mean(axis=0)
        self.p0_mean = p0.mean(axis=0)

        if fit_scale:
            self.p_scale = np.sqrt(np.mean(np.sum((p - self.p_mean) ** 2, axis=1)))
            self.p0_scale = np.sqrt(np.mean(np.sum((p0 - self.p0_mean) ** 2, axis=1)))

            if self.p_scale <= 0 or self.p0_scale <= 0:
                raise ValueError("Cannot fit scale for degenerate point clouds.")
        else:
            self.p_scale = 1.0
            self.p0_scale = 1.0

        self.x = (p - self.p_mean) / self.p_scale
        self.x0 = (p0 - self.p0_mean) / self.p0_scale

        self.x0_sq = np.sum(self.x0 * self.x0, axis=1)[None, :]

        self.n = len(self.x)
        self.m = len(self.x0)
        self.d = self.x.shape[1]

        self.loga = -np.log(self.n)
        self.logb = -np.log(self.m)

        self.f = np.zeros(self.n)
        self.g = np.zeros(self.m)

        self.R = np.eye(self.d)
        self.s = 1.0
        self.t = np.zeros(self.d)

    def _transform_x(self):
        return self.s * (self.x @ self.R.T) + self.t

    def _compute_cost_matrix(self):
        x_t = self._transform_x()
        x_t_sq = np.sum(x_t * x_t, axis=1)[:, None]
        C = x_t_sq + self.x0_sq - 2.0 * (x_t @ self.x0.T)
        return np.maximum(C, 0.0)

    def _log_sinkhorn_step(self, cost_matrix, eps, num_sinkhorn_iterations):
        self.f, self.g = self._log_sinkhorn_run(
            self.loga,
            self.logb,
            self.f,
            self.g,
            cost_matrix,
            eps,
            num_sinkhorn_iterations,
        )

    def _sinkhorn_weights_from_cost(self, cost_matrix, eps):
        return np.exp((self.f[:, None] + self.g[None, :] - cost_matrix) / eps)

    def _attention_weights_from_cost(self, cost_matrix, eps):
        if self.attention_direction == "p_to_p0":
            # each p_i attends over all p0_j
            logw = -cost_matrix / eps
            logw = logw - logsumexp(logw, axis=1, keepdims=True)
            weights = np.exp(logw) / self.n

        elif self.attention_direction == "p0_to_p":
            # each p0_j attends over all p_i
            logw = -cost_matrix / eps
            logw = logw - logsumexp(logw, axis=0, keepdims=True)
            weights = np.exp(logw) / self.m

        return weights

    def _weights_from_cost(self, cost_matrix, eps):
        if self.coupling == "sinkhorn":
            return self._sinkhorn_weights_from_cost(cost_matrix, eps)

        if self.coupling == "attention":
            return self._attention_weights_from_cost(cost_matrix, eps)

        raise RuntimeError("Unknown coupling mode.")

    def _matching_step(self, cost_matrix, eps, num_sinkhorn_iterations):
        if self.coupling == "sinkhorn":
            self._log_sinkhorn_step(cost_matrix, eps, num_sinkhorn_iterations)

        # In attention mode there is no stateful matching step.
        # The weights are obtained directly from softmax(-C / eps).

    def _procrustes_step(self, cost_matrix, eps):
        weights = self._weights_from_cost(cost_matrix, eps)

        total_mass = np.sum(weights)
        if total_mass <= 0:
            raise RuntimeError("Weight matrix has zero total mass.")

        row_mass = np.sum(weights, axis=1)
        col_mass = np.sum(weights, axis=0)

        mu_x = np.sum(row_mass[:, None] * self.x, axis=0) / total_mass
        mu_y = np.sum(col_mass[:, None] * self.x0, axis=0) / total_mass

        Xc = self.x - mu_x
        Yc = self.x0 - mu_y

        cross_cov = Yc.T @ weights.T @ Xc

        U, S, Vt = np.linalg.svd(cross_cov, full_matrices=False)
        R = U @ Vt

        if not self.allow_reflection:
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1.0
                R = U @ Vt

        den = np.sum(row_mass[:, None] * (Xc ** 2))

        if den <= 0:
            raise RuntimeError("Degenerate weighted source point cloud.")

        if self.fit_scale:
            X_rot = Xc @ R.T
            num = np.sum(weights * (X_rot @ Yc.T))
            s = num / den
            s = max(s, self.min_scale)
        else:
            s = 1.0

        t = mu_y - s * (mu_x @ R.T)

        self.R = R
        self.s = s
        self.t = t

    def _final_cost_and_weights(self, eps, num_sinkhorn_iterations):
        cost_matrix = self._compute_cost_matrix()

        if self.coupling == "sinkhorn":
            self._log_sinkhorn_step(cost_matrix, eps, num_sinkhorn_iterations)

        weights = self._weights_from_cost(cost_matrix, eps)
        return cost_matrix, weights

    def run(
        self,
        eps_start,
        eps_end,
        num_sinkhorn_iterations=20,
        num_steps=50,
        eps_log_decay=True,
        return_score=False,
        return_weight_matrix=False,
        verbose=True,
    ):
        if eps_start <= 0 or eps_end <= 0:
            raise ValueError("eps_start and eps_end must be positive.")

        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        if num_sinkhorn_iterations <= 0:
            raise ValueError("num_sinkhorn_iterations must be positive.")

        if eps_log_decay:
            eps_values = np.geomspace(eps_start, eps_end, num_steps)
        else:
            eps_values = np.linspace(eps_start, eps_end, num_steps)

        iterator = tqdm(eps_values) if verbose else eps_values

        for eps in iterator:
            cost_matrix = self._compute_cost_matrix()
            self._matching_step(cost_matrix, eps, num_sinkhorn_iterations)
            self._procrustes_step(cost_matrix, eps)

        cost_matrix, weights = self._final_cost_and_weights(
            eps_end,
            num_sinkhorn_iterations,
        )

        score = np.sum(weights * cost_matrix)

        s_true = self.s * self.p0_scale / self.p_scale
        t_true = (
            self.p0_mean
            + self.p0_scale * self.t
            - s_true * (self.p_mean @ self.R.T)
        )

        return_list = [self.R, s_true, t_true]

        if return_score:
            return_list.append(score)

        if return_weight_matrix:
            return_list.append(weights)

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
        points = np.asarray(points, dtype=float)
        return s * (points @ R.T) + t

    @staticmethod
    def inverse_transform(points, R, s, t):
        points = np.asarray(points, dtype=float)
        return (points - t) @ R / s