import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMomentsConvNd(nn.Module):
    """
    Gaussian local moments in 2D or 3D.

    For ndim=2:
        input: (B, C, H, W)

    For ndim=3:
        input: (B, C, D, H, W)

    Outputs:
        m0:     (B, S, C, *spatial)
        m1:     (B, S, ndim, C, *spatial)
        m2:     (B, S, ndim, ndim, C, *spatial)
        valid:  (B, S, C, *spatial)

    If normalize=True:
        mu:     (B, S, ndim, C, *spatial)
        second: (B, S, ndim, ndim, C, *spatial)
        cov:    (B, S, ndim, ndim, C, *spatial)
    """

    def __init__(
        self,
        num_channels,
        sigmas,
        spacings=1.0,
        ndim=2,
        cutoff_factor=3.0,
        padding_mode="replicate",
        eps=1e-8,
        min_mass=None,
        density_mode="clamp",
        cov_floor=0.0,
        compute_dtype=torch.float32,
    ):
        super().__init__()

        if ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3.")

        if density_mode not in ("clamp", "raw", "abs", "softplus", "square"):
            raise ValueError(
                "density_mode must be one of: "
                "'clamp', 'raw', 'abs', 'softplus', 'square'."
            )

        self.num_channels = int(num_channels)
        self.ndim = int(ndim)
        self.cutoff_factor = float(cutoff_factor)
        self.padding_mode = padding_mode
        self.eps = float(eps)
        self.min_mass = float(eps if min_mass is None else min_mass)
        self.density_mode = density_mode
        self.cov_floor = float(cov_floor)
        self.compute_dtype = compute_dtype

        sigmas = torch.as_tensor(sigmas, dtype=torch.float32).flatten()

        if sigmas.numel() < 1:
            raise ValueError("sigmas must contain at least one scale.")

        if torch.any(sigmas <= 0):
            raise ValueError("All sigmas must be positive.")

        if isinstance(spacings, (float, int)):
            spacings = [float(spacings)] * ndim

        spacings = torch.as_tensor(spacings, dtype=torch.float32).flatten()

        if spacings.numel() != ndim:
            raise ValueError(f"spacings must be scalar or length {ndim}.")

        if torch.any(spacings <= 0):
            raise ValueError("All spacings must be positive.")

        self.register_buffer("sigmas", sigmas)
        self.register_buffer("spacings", spacings)

        half_conv_dims = torch.ceil(
            self.cutoff_factor * sigmas[:, None] / spacings[None, :]
        ).to(torch.long)

        self.register_buffer("half_conv_dims", half_conv_dims)

        for s in range(sigmas.numel()):
            sigma = float(sigmas[s].item())

            half_dims = [
                int(half_conv_dims[s, ax].item())
                for ax in range(ndim)
            ]

            spacing_vals = [
                float(spacings[ax].item())
                for ax in range(ndim)
            ]

            coords_1d = [
                torch.arange(-h, h + 1, dtype=torch.float32) * dx
                for h, dx in zip(half_dims, spacing_vals)
            ]

            grids = torch.meshgrid(*coords_1d, indexing="ij")
            r2 = sum(g * g for g in grids)

            G = torch.exp(-r2 / (2.0 * sigma * sigma))
            G = G / G.sum()

            k0 = G[None]

            k1 = torch.stack(
                [G * grids[a] for a in range(ndim)],
                dim=0,
            )

            k2 = torch.stack(
                [
                    torch.stack(
                        [G * grids[a] * grids[b] for b in range(ndim)],
                        dim=0,
                    )
                    for a in range(ndim)
                ],
                dim=0,
            )

            self.register_buffer(f"k0_{s}", k0)
            self.register_buffer(f"k1_{s}", k1)
            self.register_buffer(f"k2_{s}", k2)

    def _make_density(self, x):
        if self.compute_dtype is not None:
            x = x.to(dtype=self.compute_dtype)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.density_mode == "raw":
            return x
        if self.density_mode == "clamp":
            return x.clamp_min(0.0)
        if self.density_mode == "abs":
            return x.abs()
        if self.density_mode == "softplus":
            return F.softplus(x)
        if self.density_mode == "square":
            return x * x

        raise RuntimeError("Unknown density_mode.")

    def _pad_tuple(self, kernel_shape):
        pad = []

        for k in reversed(kernel_shape):
            h = k // 2
            pad.extend([h, h])

        return tuple(pad)

    def _depthwise_conv(self, x, kernel):
        """
        kernel:
            (K, *kernel_shape)

        returns:
            (B, K, C, *spatial)
        """

        K = kernel.shape[0]
        kernel_shape = kernel.shape[1:]

        kernel = kernel.to(dtype=x.dtype, device=x.device)

        weight = kernel[:, None]

        weight = weight[None].expand(
            self.num_channels,
            K,
            1,
            *kernel_shape,
        )

        weight = weight.reshape(
            self.num_channels * K,
            1,
            *kernel_shape,
        )

        x_pad = F.pad(x, self._pad_tuple(kernel_shape), mode=self.padding_mode)

        if self.ndim == 2:
            y = F.conv2d(
                x_pad,
                weight,
                bias=None,
                stride=1,
                padding=0,
                groups=self.num_channels,
            )
        else:
            y = F.conv3d(
                x_pad,
                weight,
                bias=None,
                stride=1,
                padding=0,
                groups=self.num_channels,
            )

        B = x.shape[0]
        spatial = y.shape[2:]

        y = y.reshape(B, self.num_channels, K, *spatial)
        y = y.permute(0, 2, 1, *range(3, 3 + self.ndim)).contiguous()

        return y

    def _valid_from_m0(self, m0):
        if self.density_mode == "raw":
            return m0.abs() > self.min_mass

        return m0 > self.min_mass

    def _safe_divide_and_zero(self, numerator, m0, valid, num_tensor_axes):
        """
        numerator:
            (B, S, tensor_axes..., C, *spatial)

        m0:
            (B, S, C, *spatial)

        valid:
            (B, S, C, *spatial)
        """

        denom = torch.where(valid, m0, torch.ones_like(m0))
        mask = valid

        for _ in range(num_tensor_axes):
            denom = denom.unsqueeze(2)
            mask = mask.unsqueeze(2)

        out = numerator / denom
        out = torch.where(mask, out, torch.zeros_like(out))
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out

    def _add_cov_floor_only_valid(self, cov, valid):
        """
        cov:
            (B, S, ndim, ndim, C, *spatial)

        valid:
            (B, S, C, *spatial)
        """

        valid_cov = valid[:, :, None, None]

        cov = torch.where(valid_cov, cov, torch.zeros_like(cov))

        if self.cov_floor > 0:
            eye = torch.eye(self.ndim, dtype=cov.dtype, device=cov.device)
            eye_shape = (1, 1, self.ndim, self.ndim, 1) + (1,) * self.ndim
            eye = eye.reshape(eye_shape)

            cov_floored = cov + self.cov_floor * eye

            # Invalid regions stay exactly zero.
            cov = torch.where(valid_cov, cov_floored, torch.zeros_like(cov))

        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

        return cov

    def forward(self, x, normalize=True, central_cov=True):
        if x.ndim != self.ndim + 2:
            raise ValueError(
                f"Expected input with {self.ndim + 2} dims, got {x.shape}."
            )

        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {x.shape[1]}."
            )

        x_eff = self._make_density(x)

        all_m0 = []
        all_m1 = []
        all_m2 = []

        B = x_eff.shape[0]
        spatial = x_eff.shape[2:]

        for s in range(self.sigmas.numel()):
            k0 = getattr(self, f"k0_{s}")
            k1 = getattr(self, f"k1_{s}")
            k2 = getattr(self, f"k2_{s}")

            # (B, C, *spatial)
            m0_s = self._depthwise_conv(x_eff, k0)[:, 0]

            # (B, ndim, C, *spatial)
            m1_s = self._depthwise_conv(x_eff, k1)

            # (B, ndim * ndim, C, *spatial)
            k2_flat = k2.reshape(self.ndim * self.ndim, *k2.shape[2:])
            m2_s_flat = self._depthwise_conv(x_eff, k2_flat)

            # (B, ndim, ndim, C, *spatial)
            m2_s = m2_s_flat.reshape(
                B,
                self.ndim,
                self.ndim,
                self.num_channels,
                *spatial,
            )

            all_m0.append(m0_s)
            all_m1.append(m1_s)
            all_m2.append(m2_s)

        m0 = torch.stack(all_m0, dim=1)
        m1 = torch.stack(all_m1, dim=1)
        m2 = torch.stack(all_m2, dim=1)

        m0 = torch.nan_to_num(m0, nan=0.0, posinf=0.0, neginf=0.0)
        m1 = torch.nan_to_num(m1, nan=0.0, posinf=0.0, neginf=0.0)
        m2 = torch.nan_to_num(m2, nan=0.0, posinf=0.0, neginf=0.0)

        valid = self._valid_from_m0(m0)

        # Zero raw tensor moments where local density is invalid.
        m1 = torch.where(valid[:, :, None], m1, torch.zeros_like(m1))
        m2 = torch.where(valid[:, :, None, None], m2, torch.zeros_like(m2))

        out = {
            "m0": m0,
            "m1": m1,
            "m2": m2,
            "valid": valid,
        }

        if normalize:
            mu = self._safe_divide_and_zero(
                numerator=m1,
                m0=m0,
                valid=valid,
                num_tensor_axes=1,
            )

            second = self._safe_divide_and_zero(
                numerator=m2,
                m0=m0,
                valid=valid,
                num_tensor_axes=2,
            )

            out["mu"] = mu
            out["second"] = second

            if central_cov:
                cov = second - mu[:, :, :, None] * mu[:, :, None, :]

                # Symmetrize tensor indices.
                cov = 0.5 * (cov + cov.transpose(2, 3))

                # Invalid regions are zero. Optional diagonal floor only on valid regions.
                cov = self._add_cov_floor_only_valid(cov, valid)

                out["cov"] = cov

        return out


def anisotropy_q_from_cov_2d(cov, valid=None, channel=0, eps=1e-8):
    """
    cov shape:
        (B, S, 2, 2, C, H, W)

    valid shape:
        None or (B, S, C, H, W)

    returns:
        q: (B, S, H, W)

    Invalid locations are exactly q = 0.
    """

    if cov.shape[2] != 2 or cov.shape[3] != 2:
        raise ValueError("anisotropy_q_from_cov_2d expects a 2D covariance.")

    C = cov[:, :, :, :, channel]  # (B, S, 2, 2, H, W)

    C = C.permute(0, 1, 4, 5, 2, 3).contiguous()  # (B, S, H, W, 2, 2)

    C = 0.5 * (C + C.transpose(-1, -2))
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    evals = torch.linalg.eigvalsh(C)

    vals_n = evals[..., 0].clamp_min(eps)
    vals_p = evals[..., 1].clamp_min(eps)

    q = 0.25 * torch.log(vals_p / vals_n)
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    if valid is not None:
        valid_c = valid[:, :, channel]
        q = torch.where(valid_c, q, torch.zeros_like(q))

    return q