import numpy as np
import torch as torch
import torch.fft as fft
import argparse
from tqdm import tqdm
from math import sqrt
import torch.nn.functional as F

def interp_periodic_2d(field, x, y, dx, dy):
    Nx = field.shape[0]
    Ny = field.shape[1]
    xi = x / dx
    yi = y / dy
    i0 = int(torch.floor(xi)) % Nx
    j0 = int(torch.floor(yi)) % Ny
    w_x = xi - torch.floor(xi)
    w_y = yi - torch.floor(yi)

    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    return (1.0 - w_x) * (1.0 - w_y) * field[i0, j0] + \
           w_x * (1.0 - w_y) * field[i1, j0] + \
           (1.0 - w_x) * w_y * field[i0, j1] + \
           w_x * w_y * field[i1, j1]


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
    parser.add_argument("--dx", type=float, default=0.01, help="Spatial grid spacing (x)")
    parser.add_argument("--dy", type=float, default=0.01, help="Spatial grid spacing (y)")
    parser.add_argument("--Nx", type=int, default=16384, help="Number of grid points (x)")
    parser.add_argument("--Ny", type=int, default=1024, help="Number of grid points (y)")
    # seconds
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step for the simulation")
    # micrometers
    parser.add_argument("--field_length_scale", type=float, default=0.05, help="Characteristic length scale of the field")
    # seconds
    parser.add_argument("--field_time_scale", type=float, default=1.0, help="Characteristic time scale of the field")
    # seconds
    parser.add_argument("--particle_time_scale", type=float, default=0.1, help="Characteristic time scale of the particle")
    # E = mu * phi^2 / 2 * l  -> E= mu/l -> mu = E*l = F*l^2 
    parser.add_argument("--mu", type=float, default=1.0, help="Field mass parameter")
    # pN/um
    parser.add_argument("--kappa", type=float, default=2.5, help="Particle trap stiffness")
    parser.add_argument("--coupling", type=float, default=0.1, help="Coupling strength between particle and field")
    parser.add_argument("--r0", type=float, default=1.0, help="Interaction range")
    parser.add_argument("--vx", type=float, default=1.0, help="Trap velocity (x)")
    parser.add_argument("--vy", type=float, default=0.0, help="Trap velocity (y)")
    parser.add_argument("--temp", type=float, default=0.004, help="Temperature in units of energy")
    parser.add_argument("--num_cycles", type=int, default=0, help="Number of simulation steps in units of trap cycles (overrides sim_time)")
    parser.add_argument("--sim_time", type=float, default=1000.0, help="Total simulation time (overrides num_cycles)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_snapshots", type=int, default=1000, help="Number of snapshots to save for the field configuration")
    parser.add_argument("--verbosity", type=int, default=1000, help="Print progress every N steps")
    args = parser.parse_args()

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running simulation on device: {device}")

    # ---- parameters of the simulation ----
    dx = args.dx
    dy = args.dy
    Nx = args.Nx
    Ny = args.Ny
    Lx = Nx * dx
    Ly = Ny * dy
    x = torch.arange(Nx) * dx
    y = torch.arange(Ny) * dy
    qx = 2*torch.pi*fft.fftfreq(Nx, d=dx)
    qy = 2*torch.pi*fft.fftfreq(Ny, d=dy)
    q2 = qx[:, torch.newaxis]**2 + qy[torch.newaxis, :]**2
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
    vx = args.vx
    vy = args.vy
    v = torch.tensor([vx, vy])

    # ---- machinery ----
    lambda_z_dt = torch.ones((2,)) * kappa*dt/gamma
    lambda_q_dt = (mu + K*q2)*(dt*q2) / gamma_tilde
    exp_integrator_z = torch.exp(-lambda_z_dt)
    exp_integrator_q = torch.exp(-lambda_q_dt)
    def integrator_negated(x):
        safe_x = torch.where(torch.abs(x) < 1e-10, 1.0, x)
        return torch.where(torch.abs(x) < 1e-10, 1.0, (1 - torch.exp(-safe_x)) / safe_x)

    integration_factor_z = dt*integrator_negated(lambda_z_dt)
    integration_factor_q = dt*integrator_negated(lambda_q_dt)
    noise_sigma_z  = torch.sqrt(integrator_negated(2*lambda_z_dt)*2*kbT*dt/gamma)
    noise_sigma_q  = torch.sqrt(integrator_negated(2*lambda_q_dt)*2*q2*kbT*dt/gamma_tilde)
    V_xy = torch.exp(-0.5*(x[:, torch.newaxis]**2+y[torch.newaxis, :]**2)/r0**2)
    V_xy *= 1/(torch.sum(V_xy)*dx*dy)
    V_q = fft.fft2(V_xy, dim=(0,1))
    grad_kernel_qx = 1j*qx[:, torch.newaxis] * torch.conj(V_q)
    grad_kernel_qy = 1j*qy[torch.newaxis, :] * torch.conj(V_q)
    grad_kernel_q = torch.stack((grad_kernel_qx, grad_kernel_qy))
    interaction_term_field = (coupling/gamma_tilde)*q2*V_q


    v = v.to(device)
    x = x.to(device)
    y = y.to(device)
    qx = qx.to(device)
    qy = qy.to(device)
    exp_integrator_q = exp_integrator_q.to(device)
    integration_factor_q = integration_factor_q.to(device)
    noise_sigma_q = noise_sigma_q.to(device)
    exp_integrator_z = exp_integrator_z.to(device)
    integration_factor_z = integration_factor_z.to(device)
    noise_sigma_z = noise_sigma_z.to(device)
    grad_kernel_q = grad_kernel_q.to(device)
    interaction_term_field = interaction_term_field.to(device)
    

    
    torch.manual_seed(args.seed)
    n_equilibration_steps = int(20*field_time_scale/dt)
    phi_q = torch.randn(Nx, Ny)
    phi_q -= torch.mean(phi_q)
    phi_q = fft.fft2(phi_q, dim=(0,1))
    phi_q[0, 0] = 0.0
    phi_q = phi_q.to(device)
    pbar = tqdm(range(n_equilibration_steps), disable=(args.verbosity==0))
    for step in  pbar:
        if args.verbosity > 0 and step % args.verbosity == 0:
            pbar.set_description(f"Step (equilibration) {step}/{n_equilibration_steps}")
        noise_q = noise_sigma_q * fft.fft2(torch.randn(Nx, Ny, device=device), dim=(0,1)) / sqrt(Nx*Ny)
        phi_q = exp_integrator_q*phi_q + noise_q

    if args.sim_time > 0:
        n_simulation_steps = int(args.sim_time / dt)
    elif args.num_cycles > 0:
        if abs(vx) > 0 and abs(vy) > 0:
            n_simulation_steps = int(args.num_cycles * max(Lx/abs(vx), Ly/abs(vy)) / dt)
        elif abs(vx) > 0:
            n_simulation_steps = int(args.num_cycles * Lx / (abs(vx)*dt))
        elif abs(vy) > 0:
            n_simulation_steps = int(args.num_cycles * Ly / (abs(vy)*dt))
        else:
            n_simulation_steps = args.num_cycles * int(10*max(particle_time_scale, field_time_scale)/dt)
    else:
        n_simulation_steps = max(1, int(10*particle_time_scale/dt))
    
    save_phi_every = max(1, n_simulation_steps//args.num_snapshots)

    # state
    
    trap_origin = torch.zeros((2,))
    if abs(vx) > 0 and abs(vy) > 0:
        trap_origin = torch.tensor([Lx/2, Ly/2])
    elif abs(vx) > 0:
        trap_origin = torch.tensor([0, Ly/2])
    elif abs(vy) > 0:
        trap_origin = torch.tensor([Lx/2, 0])
    else:
        trap_origin = torch.tensor([Lx/2, Ly/2])
    trap_origin = trap_origin.to(device)
    z = trap_origin.clone()
    z = z.to(device)
    z_wrapped = z % torch.tensor([Lx, Ly], device=device)
    
    z_wrapped = z_wrapped.to(device)
    
    z_hist = np.zeros((n_simulation_steps, 2))
    pbar = tqdm(range(n_simulation_steps), disable=(args.verbosity==0))


    redraw_noise_every = 20
    noise_q = None
    force_particle = torch.zeros((2,), device=device)
    #phi_qs = []
    for step in pbar:
        if step % redraw_noise_every == 0:
            noise_q = noise_sigma_q * fft.fft2(torch.randn(redraw_noise_every, Nx, Ny, device=device), dim=(1,2)) / sqrt(Nx*Ny)
            
        if args.verbosity > 0 and step % args.verbosity == 0:
            pbar.set_description(f"Step {step}/{n_simulation_steps}. Memory usage (CUDA): {torch.cuda.memory_allocated(device)/1e9:.2f} GB.")
        # store
        z_hist[step] = z.cpu().numpy()
        #if step % save_phi_every == 0:
        #    phi_qs.append(phi_q.cpu().numpy())
        z_wrapped = z % torch.tensor([Lx, Ly], device=device)
        
        # forces at z_n
        interaction_term_particle = fft.ifft2(grad_kernel_q * phi_q, dim=(1,2)).real
        force_particle[0] = (coupling/gamma) * interp_periodic_2d(interaction_term_particle[0], z_wrapped[0], z_wrapped[1], dx, dy)
        force_particle[1] = (coupling/gamma) * interp_periodic_2d(interaction_term_particle[1], z_wrapped[0], z_wrapped[1], dx, dy)

        force_field = interaction_term_field * torch.exp(-1j * (z_wrapped[0] * qx[:, None] + z_wrapped[1] * qy[None, :]))
        # noise
        noise_z = noise_sigma_z * torch.randn(2, device=device)
        z = exp_integrator_z*z + integration_factor_z*force_particle + noise_z
        # update field (explicit in z_n)
        phi_q = exp_integrator_q*phi_q + integration_factor_q*force_field + noise_q[step % redraw_noise_every]
        phi_q[0, 0] = 0.0
    #phi_qs = np.array(phi_qs)

    simulation_params = {
        'dx': dx,
        'Nx': Nx,
        'Ny': Ny,
        'Lx': Lx,
        'Ly': Ly,
        'dt': dt,
        'field_length_scale': field_length_scale,
        'field_time_scale': field_time_scale,
        'particle_time_scale': particle_time_scale,
        'mu': mu,
        'kappa': kappa,
        'coupling': coupling,
        'r0': r0,
        'vx': v[0].item(),
        'vy': v[1].item(),
        'kbT': kbT,
    }

    simulation_name = f"simulation_dx={dx}_dy={dy}_Nx={Nx}_Ny={Ny}_dt={dt}_field_length_scale={field_length_scale}_field_time_scale={field_time_scale}_particle_time_scale={particle_time_scale}_mu={mu}_kappa={kappa}_coupling={coupling}_r0={r0}_vx={v[0].item()}_vy={v[1].item()}_temp={kbT}"

    np.savez(f"data2d/{simulation_name}.npz", z_hist=z_hist, params=simulation_params)