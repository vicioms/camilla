"""Particle coupled to a conserved Gaussian field in a periodic box.

H = integral[mu*phi**2/2 + K*|grad(phi)|**2/2 - g*V(x-z)*phi] dx
    + kappa*|z - R(t)|**2/2, with R(t) = trap_origin + v*t.

Linear Ornstein-Uhlenbeck terms are integrated exactly. The interaction
uses the beginning-of-step particle and field, so check convergence in dt.

Outputs in each unique run directory:
    params.json, times.npy, z_hist.npy, phi_times.npy, snapshot_steps.npy,
    and phi.npy when field snapshots are requested.
The .npy arrays are preallocated and filled progressively. Read completion
information from params.json before using an interrupted run.

Directory paths identify grid, field, particle, protocol, and output settings.
An SHA-256 fingerprint covers the complete configuration and this source file.
Atomic creation of the timestamped leaf prevents reuse of an existing run.
"""

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.fft as fft
from tqdm import tqdm


def create_run_directory(root, configuration):
    """Allocate a new directory, including for identical concurrent runs.

    The fingerprint excludes destination and progress fields, so moving a
    configuration to another parent does not change its configuration ID.
    New configuration fields automatically participate in the fingerprint.
    """
    excluded = {
        "output_dir", "status", "completed_steps", "completed_snapshots",
        "configuration_sha256",
    }
    config = {k: v for k, v in configuration.items() if k not in excluded}
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    groups = (
        ("grid", ("Nx", "Ny", "dx", "dy", "dt")),
        ("field", ("field_length_scale", "field_time_scale", "mu")),
        ("particle", (
            "particle_time_scale", "kappa", "coupling", "r0", "vx", "vy", "temp",
        )),
        ("protocol", (
            "num_cycles", "total_time", "seed", "uncoupled_init",
        )),
        ("sampling", (
            "num_snapshots", "snapshot_stride", "device", "state_dtype",
        )),
    )

    # Split long components rather than relying on one enormous filename.
    components = []
    for label, keys in groups:
        component = label
        for key in keys:
            value = config[key]
            display = format(value, ".12g") if isinstance(value, float) else str(value)
            token = f"{key}={display}"
            if len((component + "__" + token).encode("utf-8")) > 180:
                components.append(component)
                component = label + "_continued"
            component += "__" + token
        components.append(component)

    parent = Path(root).expanduser().joinpath(*components, "cfg_" + fingerprint)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = Path(tempfile.mkdtemp(prefix=f"run_{timestamp}_", dir=str(parent)))
    return out, fingerprint


def get_params_from_scales(
    field_length_scale, field_time_scale, particle_time_scale, mu, kappa,
):
    return {
        "gamma": kappa * particle_time_scale,
        "K": mu * field_length_scale**2,
        "gamma_tilde": mu * field_time_scale / field_length_scale**2,
    }


def trap_cycle_time(Lx, Ly, vx, vy):
    """Return a common period of the wrapped trap trajectory."""
    periods = [
        length / abs(speed)
        for length, speed in ((Lx, vx), (Ly, vy)) if speed != 0.0
    ]
    if not periods:
        raise ValueError("num_cycles requires vx or vy to be nonzero.")
    if len(periods) == 1:
        return periods[0]
    tx, ty = periods
    ratio = Fraction(tx / ty).limit_denominator(10_000)
    period = ratio.denominator * tx
    if ratio.numerator < 1 or not math.isclose(
        period / ty, ratio.numerator, rel_tol=0.0, abs_tol=1e-8,
    ):
        raise ValueError(
            "No short common return period was found. "
            "Use --sim_time with --num_cycles 0 for a nonclosing trajectory."
        )
    return period


def snapshot_indices(n_steps, requested):
    count = min(requested, n_steps + 1)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == 1:
        return np.array([n_steps], dtype=np.int64)
    return np.rint(np.linspace(0, n_steps, count)).astype(np.int64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Particle coupled to a conserved Gaussian field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    float_defaults = {
        "dx": 0.01, "dy": 0.01, "dt": 1e-3,
        "field_length_scale": 0.05, "field_time_scale": 1.0,
        "particle_time_scale": 0.1, "mu": 1.0, "kappa": 2.5,
        "coupling": 0.1, "r0": 1.0, "vx": 1.0, "vy": 0.0, "temp": 0.004,
    }
    for name, default in float_defaults.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--Nx", type=int, default=16384)
    parser.add_argument("--Ny", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_cycles", type=int, default=0,
        help="Complete trap returns; positive values override sim_time",
    )
    parser.add_argument(
        "--sim_time", type=float, default=1000.0,
        help="Duration when num_cycles is zero",
    )
    parser.add_argument(
        "--num_snapshots", type=int, default=1000,
        help="Number of field snapshots; zero disables them",
    )
    parser.add_argument(
        "--snapshot_stride", type=int, default=1,
        help="Save every Nth grid point along each axis",
    )
    parser.add_argument(
        "--uncoupled_init", action="store_true",
        help="Start with a free equilibrium field; switch coupling on at t=0",
    )
    parser.add_argument("--verbosity", type=int, default=1000)
    parser.add_argument("--output_dir", default="data2d")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    positive = (
        "dx", "dy", "dt", "field_length_scale", "field_time_scale",
        "particle_time_scale", "mu", "kappa", "r0",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    for name in ("vx", "vy", "coupling", "temp", "sim_time"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")
    if args.Nx < 3 or args.Ny < 3:
        parser.error("Nx and Ny must both be at least 3")
    if args.temp < 0:
        parser.error("temp must be nonnegative")
    if args.num_cycles < 0 or args.num_snapshots < 0:
        parser.error("num_cycles and num_snapshots must be nonnegative")
    if args.num_cycles == 0 and args.sim_time <= 0:
        parser.error("sim_time must be positive when num_cycles is zero")
    if args.snapshot_stride < 1 or args.verbosity < 0:
        parser.error("snapshot_stride must be positive; verbosity nonnegative")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must lie in [0, 2**63)")
    return args


@torch.no_grad()
def run(args):
    Nx, Ny = args.Nx, args.Ny
    dx, dy, dt = args.dx, args.dy, args.dt
    Lx, Ly = Nx * dx, Ny * dy
    cell_area, n_grid = dx * dy, Nx * Ny
    mu, kappa = args.mu, args.kappa
    coupling, kbT = args.coupling, args.temp
    params = get_params_from_scales(
        args.field_length_scale, args.field_time_scale,
        args.particle_time_scale, mu, kappa,
    )
    K, gamma, gamma_tilde = params["K"], params["gamma"], params["gamma_tilde"]
    tau_z = gamma / kappa

    cycle_time = None
    if args.num_cycles > 0:
        cycle_time = trap_cycle_time(Lx, Ly, args.vx, args.vy)
        total_time = args.num_cycles * cycle_time
    else:
        total_time = args.sim_time
    if not math.isfinite(total_time) or total_time <= 0:
        raise ValueError("The requested simulation duration is invalid.")
    ratio = total_time / dt
    nearest = round(ratio)
    if math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-9):
        n_steps = max(1, nearest)
    else:
        n_steps = max(1, math.ceil(ratio))
    last_dt = total_time - (n_steps - 1) * dt

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    dtype = torch.float64
    torch.manual_seed(args.seed)
    box = torch.tensor([Lx, Ly], dtype=dtype, device=device)
    v = torch.tensor([args.vx, args.vy], dtype=dtype, device=device)
    if args.vx != 0 and args.vy == 0:
        origin = [0.0, Ly / 2]
    elif args.vy != 0 and args.vx == 0:
        origin = [Lx / 2, 0.0]
    else:
        origin = [Lx / 2, Ly / 2]
    trap_origin = torch.tensor(origin, dtype=dtype, device=device)

    qx = 2 * math.pi * fft.fftfreq(Nx, d=dx, dtype=dtype, device=device)
    qy = 2 * math.pi * fft.rfftfreq(Ny, d=dy, dtype=dtype, device=device)
    q2 = qx[:, None]**2 + qy[None, :]**2
    stiffness_q = mu + K * q2
    rate_q = q2 * stiffness_q / gamma_tilde
    V_q = torch.exp(-0.5 * args.r0**2 * q2) / cell_area
    # Exclude ambiguous translated Nyquist components only from the kernel.
    if Nx % 2 == 0:
        V_q[Nx // 2, :] = 0
    if Ny % 2 == 0:
        V_q[:, -1] = 0
    weights = torch.ones_like(qy)
    weights[1:] = 2
    if Ny % 2 == 0:
        weights[-1] = 1
    thermal_scale = torch.sqrt(kbT / (cell_area * stiffness_q))

    def shifted_kernel(z):
        zw = torch.remainder(z, box)
        phase = qx[:, None] * zw[0] + qy[None, :] * zw[1]
        return V_q * torch.exp(-1j * phase)

    def particle_drift(phi_q, shifted_V_q):
        overlap = weights[None, :] * (phi_q.conj() * shifted_V_q).imag
        fx = torch.dot(qx, overlap.sum(dim=1))
        fy = torch.dot(qy, overlap.sum(dim=0))
        return coupling * cell_area / (gamma * n_grid) * torch.stack((fx, fy))

    def coefficients(h):
        field_relax = -torch.expm1(-rate_q * h)
        field_noise = thermal_scale * torch.sqrt(-torch.expm1(-2 * rate_q * h))
        particle_decay = math.exp(-h / tau_z)
        particle_integral = -tau_z * math.expm1(-h / tau_z)
        particle_noise = math.sqrt((kbT / kappa) * (-math.expm1(-2 * h / tau_z)))
        return (
            field_relax, field_noise,
            particle_decay, particle_integral, particle_noise,
        )

    regular_coefficients = coefficients(dt)
    # Exact static joint equilibrium before the trap starts moving.
    u = math.sqrt(kbT / kappa) * torch.randn(2, dtype=dtype, device=device)
    z = trap_origin + u
    if kbT > 0:
        phi_q = fft.rfft2(
            torch.randn(Nx, Ny, dtype=dtype, device=device)
        ) * thermal_scale
    else:
        phi_q = torch.zeros(V_q.shape, dtype=torch.complex128, device=device)
    if not args.uncoupled_init and coupling != 0:
        phi_q += coupling * shifted_kernel(z) / stiffness_q
    phi_q[0, 0] = 0

    stride = args.snapshot_stride
    metadata = {
        **vars(args), **params,
        "Lx": Lx, "Ly": Ly, "trap_origin": origin,
        "cycle_time": cycle_time, "total_time": total_time,
        "n_steps": n_steps, "last_dt": last_dt, "field_mean": 0.0,
        "state_dtype": "float64", "snapshot_dtype": "float32",
        "snapshot_dx": stride * dx, "snapshot_dy": stride * dy,
        "snapshot_representation": "real-space phi",
        "runtime_device": str(device),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running", "completed_steps": 0, "completed_snapshots": 0,
    }
    out, config_id = create_run_directory(args.output_dir, metadata)
    metadata["configuration_sha256"] = config_id

    def write_metadata():
        (out / "params.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )

    write_metadata()
    snap_steps = snapshot_indices(n_steps, args.num_snapshots)
    snap_times = np.minimum(snap_steps.astype(float) * dt, total_time)
    np.save(out / "snapshot_steps.npy", snap_steps)
    np.save(out / "phi_times.npy", snap_times)
    z_hist = np.lib.format.open_memmap(
        out / "z_hist.npy", mode="w+", dtype=np.float64, shape=(n_steps + 1, 2),
    )
    times = np.lib.format.open_memmap(
        out / "times.npy", mode="w+", dtype=np.float64, shape=(n_steps + 1,),
    )
    snapshot_shape = ((Nx + stride - 1) // stride, (Ny + stride - 1) // stride)
    phi_store = None
    if len(snap_steps):
        phi_store = np.lib.format.open_memmap(
            out / "phi.npy", mode="w+", dtype=np.float32,
            shape=(len(snap_steps), *snapshot_shape),
        )
    expected_bytes = 24 * (n_steps + 1)
    expected_bytes += 4 * len(snap_steps) * snapshot_shape[0] * snapshot_shape[1]
    print(f"Device: {device}")
    print(f"Duration: {total_time:g}; steps: {n_steps}")
    if cycle_time is not None:
        print(f"Trap cycle time: {cycle_time:g}")
    print(f"Configuration: {config_id}")
    print(f"Output: {out.resolve()}")
    print(f"Array storage: approximately {expected_bytes / 1e9:.3f} GB")

    history_buffer = torch.empty(
        min(1024, n_steps + 1), 2, dtype=dtype, device=device,
    )
    pending, last_recorded, next_snapshot = 0, -1, 0

    def flush_history():
        nonlocal pending
        if pending:
            start = last_recorded + 1 - pending
            z_hist[start:last_recorded + 1] = history_buffer[:pending].cpu().numpy()
            pending = 0

    def record(index, t, position):
        nonlocal pending, last_recorded, next_snapshot
        history_buffer[pending].copy_(position)
        pending += 1
        last_recorded = index
        times[index] = t
        if pending == len(history_buffer):
            flush_history()
        if next_snapshot < len(snap_steps) and index == snap_steps[next_snapshot]:
            field = fft.irfft2(phi_q, s=(Nx, Ny))
            phi_store[next_snapshot] = (
                field[::stride, ::stride].to(device="cpu", dtype=torch.float32).numpy()
            )
            next_snapshot += 1

    status = "interrupted"
    try:
        record(0, 0.0, z)
        progress = tqdm(range(n_steps), disable=args.verbosity == 0)
        for step in progress:
            coeff = regular_coefficients
            if step == n_steps - 1 and last_dt != dt:
                coeff = coefficients(last_dt)
            relax_q, sigma_q, decay_z, integral_z, sigma_z = coeff
            if coupling != 0:
                V_shift = shifted_kernel(z)
                drift = particle_drift(phi_q, V_shift)
                target_phi_q = coupling * V_shift / stiffness_q
            else:
                drift = torch.zeros_like(u)
                target_phi_q = 0.0
            u.mul_(decay_z).add_(integral_z * (drift - v))
            if kbT > 0:
                u.add_(sigma_z * torch.randn(2, dtype=dtype, device=device))
            phi_q.add_(relax_q * (target_phi_q - phi_q))
            if kbT > 0:
                eta_q = fft.rfft2(torch.randn(Nx, Ny, dtype=dtype, device=device))
                phi_q.add_(sigma_q * eta_q)
            phi_q[0, 0] = 0
            index = step + 1
            t = total_time if index == n_steps else index * dt
            z = trap_origin + v * t + u
            if index % 1000 == 0 or index == n_steps:
                if not (
                    torch.isfinite(u).all().item() and torch.isfinite(phi_q).all().item()
                ):
                    raise FloatingPointError("Nonfinite state: reduce dt and check parameters.")
            record(index, t, z)
            if args.verbosity and index % args.verbosity == 0:
                progress.set_postfix(time=f"{t:.4g}")
        status = "complete"
    finally:
        flush_history()
        z_hist.flush()
        times.flush()
        if phi_store is not None:
            phi_store.flush()
        metadata["status"] = status
        metadata["completed_steps"] = max(0, last_recorded)
        metadata["completed_snapshots"] = next_snapshot
        write_metadata()
    print(f"Saved: {out.resolve()}")
    return out


if __name__ == "__main__":
    run(parse_args())
