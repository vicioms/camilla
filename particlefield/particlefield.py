import numpy as np
import numpy.fft as fft
import argparse
from tqdm import tqdm

def f_contour(f, x, r0, num_thetas):
    if np.isscalar(x):
        x = np.array([x])
    thetas = np.linspace(0, 2*np.pi, num_thetas)
    circle = r0 * np.exp(1j * thetas)
    f_values = f(x[:,None] + circle[None,:])
    return np.trapezoid(f_values, thetas, axis=1).real / (2*np.pi)

def interp_cic_1d(field, z, dx):
    N = field.size
    xi = z / dx
    j0 = int(np.floor(xi)) % N
    w = xi - np.floor(xi)

    j1 = (j0 + 1) % N

    return (1.0 - w) * field[j0] + w * field[j1]

def wrap_centered(y, L):
    return (y + L/2) % L - L/2

def unwrap_periodic(x, L):
    """Unwrap a periodic array."""
    dx = np.diff(x)
    dx_unwrapped = dx - L * np.round(dx / L)
    return np.concatenate(([x[0]], x[0] + np.cumsum(dx_unwrapped)))

def get_params_from_scales(field_length_scale, 
                           field_time_scale, 
                           particle_time_scale,
                           mu,
                           kappa):
    # K / mu = field_length_scale**2
    # gamma_tilde / mu = field_time_scale / field_length_scale**2
    # gamma / kappa = particle_trap_time_scale
    gamma = kappa * particle_time_scale
    K = mu * field_length_scale**2
    gamma_tilde = mu * field_time_scale / field_length_scale**2
    return {'gamma': gamma, 'K': K, 'gamma_tilde': gamma_tilde}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run particle-field simulation")
    # micrometers
    parser.add_argument("--dx", type=float, default=0.01, help="Spatial grid spacing")
    parser.add_argument("--N", type=int, default=4096, help="Number of grid points")
    # seconds
    parser.add_argument("--dt", type=float, default=1e-4, help="Time step for the simulation")
    # micrometers
    parser.add_argument("--field_length_scale", type=float, default=0.03, help="Characteristic length scale of the field")
    # seconds
    parser.add_argument("--field_time_scale", type=float, default=1e-1, help="Characteristic time scale of the field")
    # seconds
    parser.add_argument("--particle_time_scale", type=float, default=1e-3, help="Characteristic time scale of the particle")
    # E = mu * phi^2 / 2 * l  -> E= mu/l -> mu = E*l = F*l^2 
    parser.add_argument("--mu", type=float, default=1.0, help="Field mass parameter")
    # pN/um
    parser.add_argument("--kappa", type=float, default=50.0, help="Particle trap stiffness")
    parser.add_argument("--coupling", type=float, default=0.5, help="Coupling strength between particle and field")
    parser.add_argument("--r0", type=float, default=2.5, help="Interaction range")
    parser.add_argument("--v", type=float, default=1.0, help="Trap velocity")
    parser.add_argument("--temp", type=float, default=1e-3, help="Temperature in units of energy")
    parser.add_argument("--num_cycles", type=int, default=1, help="Number of simulation steps (default: L/(v*dt))")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_snapshots", type=int, default=1000, help="Number of snapshots to save for the field configuration")

    args = parser.parse_args()

    # ---- parameters of the simulation ----
    dx = args.dx
    N = args.N
    L = N*dx
    x = np.arange(N)*dx
    q = 2*np.pi*fft.fftfreq(N, d=dx)
    q2 = q**2
    dt = args.dt
    field_length_scale = args.field_length_scale
    field_time_scale = args.field_time_scale
    particle_time_scale = args.particle_time_scale
    mu = args.mu
    kappa = args.kappa
    kbT = args.temp
    params = get_params_from_scales(field_length_scale=field_length_scale,
                               field_time_scale=field_time_scale, 
                               particle_time_scale=particle_time_scale,
                               mu=mu,
                               kappa=kappa)
    K, gamma_tilde, gamma = params['K'], params['gamma_tilde'], params['gamma']
    
    
    # ---- interaction ----
    coupling = args.coupling
    r0 = args.r0
    v = args.v

    # ---- machinery ----
    lambda_z_dt = kappa*dt/gamma
    lambda_q_dt = (mu + K*q2)*(dt*q2) / gamma_tilde
    exp_integrator_z = np.exp(-lambda_z_dt)
    exp_integrator_q = np.exp(-lambda_q_dt)
    integrator_negated = lambda x : (1-np.exp(-x)) / x
    countour_radius = np.sqrt(5)

    integration_factor_z = dt*f_contour(integrator_negated, lambda_z_dt, countour_radius, num_thetas=512).item()
    integration_factor_q = dt*f_contour(integrator_negated, lambda_q_dt, countour_radius, num_thetas=512)
    noise_sigma_z  = np.sqrt(f_contour(integrator_negated, 2*lambda_z_dt, countour_radius, num_thetas=512).item()*2*kbT*dt/gamma)
    noise_sigma_q  = np.sqrt(f_contour(integrator_negated, 2*lambda_q_dt, countour_radius, num_thetas=512)*2*q2*kbT*dt/gamma_tilde)
    V_x = np.exp(-0.5*(x**2)/r0**2)
    V_x *= 1/(np.sum(V_x)*dx)
    V_q = fft.fft(V_x)
    grad_kernel_q = 1j*q*np.conjugate(V_q)
    interaction_term_field = (coupling/gamma_tilde)*q2*V_q

    
    np.random.seed(args.seed)
    n_equilibration_steps = int(L/(field_time_scale))
    phi_q = np.random.randn(N)
    phi_q -= np.mean(phi_q)
    phi_q = fft.fft(phi_q)
    phi_q[0] = 0.0
    for step in  tqdm(range(n_equilibration_steps)):
        noise_q = noise_sigma_q * fft.fft(np.random.randn(N)) / np.sqrt(N)
        phi_q = exp_integrator_q*phi_q + noise_q


    n_simulation_steps = args.num_cycles * int(L/(v*dt))

    save_phi_every = max(1, n_simulation_steps//args.num_snapshots)

    # state
    z = 0.0

    phi_qs = []
    z_hist = []

    for step in tqdm(range(n_simulation_steps)):
        # store
        z_hist.append(z)
        if step % save_phi_every == 0:
            phi_qs.append(phi_q.copy())

        # forces at z_n
        interaction_term_particle = fft.ifft(grad_kernel_q * phi_q).real
        force_particle = (coupling/gamma) * interp_cic_1d(interaction_term_particle, z, dx)
        force_particle += ((v*dt*step) % L)*kappa/gamma

        force_field = interaction_term_field * np.exp(-1j * z * q)

        # noises
        noise_q = noise_sigma_q * fft.fft(np.random.randn(N)) / np.sqrt(N)
        noise_z = noise_sigma_z * np.random.randn()

        z = exp_integrator_z*z + integration_factor_z*force_particle + noise_z
        z = z % L

        # update field (explicit in z_n)
        phi_q = exp_integrator_q*phi_q + integration_factor_q*force_field + noise_q
        phi_q[0] = 0.0
    phi_qs = np.array(phi_qs)
    z_hist = np.array(z_hist)

    simulation_params = {
        'dx': dx,
        'N': N,
        'L': L,
        'dt': dt,
        'field_length_scale': field_length_scale,
        'field_time_scale': field_time_scale,
        'particle_time_scale': particle_time_scale,
        'mu': mu,
        'kappa': kappa,
        'coupling': coupling,
        'r0': r0,
        'v': v,
        'kbT': kbT,
    }

    simulation_name = f"simulation_dx{dx}_N{N}_dt{dt}_field_length_scale{field_length_scale}_field_time_scale{field_time_scale}_particle_time_scale{particle_time_scale}_mu{mu}_kappa{kappa}_coupling{coupling}_r0{r0}_v{v}_temp{kbT}_num_cycles{args.num_cycles}"

    np.savez(f"data/{simulation_name}.npz", phi_qs=phi_qs, z_hist=z_hist, x=x, q=q, params=simulation_params)