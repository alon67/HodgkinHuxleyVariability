"""
Core simulation module for the JAX-optimized spatial Hodgkin-Huxley cable model.

Contains the JAX-compiled simulation functions for running the cable model.
"""

# Configure XLA environment BEFORE importing JAX
from config import configure_xla_environment
configure_xla_environment()

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, pmap
import diffrax
from functools import partial

# Import configuration
from config import (
    N_SEGMENTS, DX, LAMBDA_CABLE, INJECTION_SEGMENT, RECORDING_SEGMENT,
    T_START, T_END, RAMP_T, A_RADIUS
)

# Set JAX to use CPU and enable 64-bit precision
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


def find_equilibrium_jax(params):
    """
    Solve for an equilibrium (resting) potential V_rest by forcing net ionic current = 0
    with gating variables at their steady-state values.
    Uses Newton's method implemented in JAX for JIT compilation.

    For spatial cable model, returns same initial conditions for all N segments.
    """
    # Unpack parameters
    (gbar_Na, E_Na, A_am, B_am, C_am, A_bm, D_bm, A_ah, D_ah,
     E_bh, F_bh, G_bh,
     gbar_K, E_K, A_alpha, V_alpha, k_alpha, A_beta, tau_beta,
     C_m, E_l, G_l) = params

    # Define gating functions using JAX operations
    def alpha_m(V):
        denom = 1.0 - jnp.exp(-(V + B_am) / C_am)
        # Use where to handle division safely
        return jnp.where(jnp.abs(denom) > 1e-9, A_am * (V + B_am) / denom, A_am * C_am)

    def beta_m(V):
        V_safe = jnp.clip(V, -150.0, 100.0)
        return A_bm * jnp.exp(-V_safe / D_bm)

    def alpha_h(V):
        V_safe = jnp.clip(V, -150.0, 100.0)
        return A_ah * jnp.exp(-V_safe / D_ah)

    def beta_h(V):
        return E_bh / (1.0 + jnp.exp(-(V + F_bh) / G_bh))

    def alpha_n(V):
        V_safe = jnp.clip(V, -150.0, 100.0)
        denom = 1.0 - jnp.exp(-(V_alpha + V_safe) / k_alpha)
        return jnp.where(jnp.abs(denom) > 1e-9, A_alpha * (V_alpha + V_safe) / denom, A_alpha * k_alpha)

    def beta_n(V):
        V_safe = jnp.clip(V, -150.0, 100.0)
        return A_beta * jnp.exp(-V_safe / tau_beta)

    # Steady-state gating
    def m_inf(V):
        return alpha_m(V) / (alpha_m(V) + beta_m(V))

    def h_inf(V):
        return alpha_h(V) / (alpha_h(V) + beta_h(V))

    def n_inf(V):
        return alpha_n(V) / (alpha_n(V) + beta_n(V))

    # Net current at steady-state gating
    def net_current(V):
        mm = m_inf(V)
        hh = h_inf(V)
        nn = n_inf(V)
        I_Na = gbar_Na * mm**3 * hh * (V - E_Na)
        I_K  = gbar_K  * nn**4       * (V - E_K)
        I_L  = G_l * (V - E_l)
        return I_Na + I_K + I_L

    # Robust Newton's method with bounds and damping
    V = -65.0  # Initial guess
    V_min = -100.0  # Lower bound
    V_max = 50.0    # Upper bound

    for i in range(100):  # Increased iterations for better convergence
        f = net_current(V)

        # Numerical derivative with small step
        h_step = 1e-7
        df = (net_current(V + h_step) - f) / h_step

        # Prevent division by zero or very small derivatives
        df = jnp.where(jnp.abs(df) < 1e-10, 1e-10, df)

        # Newton step with adaptive damping
        dV = -f / df
        # Limit step size to prevent overshooting
        dV = jnp.clip(dV, -5.0, 5.0)  # Smaller steps for stability

        V_new = V + 0.3 * dV  # Stronger damping (0.3 instead of 0.5)

        # Enforce bounds
        V_new = jnp.clip(V_new, V_min, V_max)

        # Check for convergence
        change = jnp.abs(V_new - V)
        V = V_new

        # Early exit if converged (but keep iterating for JIT)
        V = jnp.where(change < 1e-8, V, V)  # Tighter tolerance

    V_rest = V
    m0 = m_inf(V_rest)
    h0 = h_inf(V_rest)
    n0 = n_inf(V_rest)

    # Return initial conditions for all segments (uniform across axon)
    V0_all = jnp.tile(V_rest, N_SEGMENTS)
    m0_all = jnp.tile(m0, N_SEGMENTS)
    h0_all = jnp.tile(h0, N_SEGMENTS)
    n0_all = jnp.tile(n0, N_SEGMENTS)

    return V0_all, m0_all, h0_all, n0_all


def simulate_HH_AP_jax(params, init_conditions, tmax=100.0, dt=0.005, I_amplitude=10.0):
    """
    Simulate the spatially-extended Hodgkin-Huxley cable equation using diffrax PDE solver.
    JAX-compatible and JIT-compiled for performance.

    Cable geometry: 10 cm length, 0.5 mm diameter, 80 segments
    Stimulus applied at injection site, voltage recorded at injection and distal sites
    Uses adaptive Tsit5 solver for accuracy with stiff cable equations

    **Memory-efficient version: Only returns voltage at 2 key locations**

    Parameters:
    -----------
    params : tuple
        Hodgkin-Huxley parameters
    init_conditions : tuple
        Initial conditions (V0_all, m0_all, h0_all, n0_all)
    tmax : float
        Simulation duration (ms)
    dt : float
        Initial timestep hint (ms)
    I_amplitude : float
        Total current injection amplitude in μA

    Returns:
    --------
    times : array
        Time vector
    V_injection : array
        Voltage at injection site
    V_distal : array
        Voltage at distal recording site
    """
    (gbar_Na, E_Na, A_am, B_am, C_am, A_bm, D_bm, A_ah, D_ah,
     E_bh, F_bh, G_bh,
     gbar_K, E_K, A_alpha, V_alpha, k_alpha, A_beta, tau_beta,
     C_m, E_l, G_l) = params

    # Gating functions for Na (m,h) using JAX operations
    def alpha_m(V):
        denom = 1.0 - jnp.exp(-(V + B_am) / C_am)
        return jnp.where(jnp.abs(denom) > 1e-9, A_am*(V + B_am)/denom, A_am*C_am)

    def beta_m(V):
        V_safe = jnp.clip(V, -150.0, 100.0)
        return A_bm * jnp.exp(-V_safe / D_bm)

    def alpha_h(V):
        V_safe = jnp.clip(V, -150.0, 100.0)
        return A_ah * jnp.exp(-V_safe / D_ah)

    def beta_h(V):
        return E_bh / (1.0 + jnp.exp(-(V + F_bh)/G_bh))

    # K gating
    def alpha_n(V):
        V_safe = jnp.clip(V, -150.0, 100.0)  # Prevent exp overflow
        denom = 1.0 - jnp.exp(-(V_alpha + V_safe) / k_alpha)
        return jnp.where(jnp.abs(denom) > 1e-9, A_alpha*(V_alpha + V_safe)/denom, A_alpha*k_alpha)

    def beta_n(V):
        V_safe = jnp.clip(V, -150.0, 100.0)  # Prevent exp overflow
        return A_beta * jnp.exp(-V_safe / tau_beta)

    # Cable equation PDE system for diffrax with spatial coupling
    def cable_hh_dynamics(t, y, args):
        I_amp = args
        N = N_SEGMENTS

        # Extract state variables for all segments
        # State vector: [V[0:N], m[N:2N], h[2N:3N], n[3N:4N]]
        V = y[0:N]
        m = y[N:2*N]
        h = y[2*N:3*N]
        n = y[3*N:4*N]

        # Clip state variables to prevent numerical instability
        # Clip V below E_Na to give the constraint enforcement room to work
        V = jnp.clip(V, -150.0, E_Na)
        m = jnp.clip(m, 0.0, 1.0)
        h = jnp.clip(h, 0.0, 1.0)
        n = jnp.clip(n, 0.0, 1.0)

        # Calculate spatial derivative (second derivative for diffusion)
        # d²V/dx² using finite differences
        d2Vdx2 = jnp.zeros(N)

        # Interior points: central difference
        d2Vdx2 = d2Vdx2.at[1:N-1].set(
            (V[2:N] - 2*V[1:N-1] + V[0:N-2]) / (DX**2)
        )

        # Boundary conditions: sealed ends (no-flux boundary, dV/dx = 0)
        # For no-flux BC, use ghost point approach: V[-1]=V[1] and V[N]=V[N-2]
        # Left boundary: d²V/dx²[0] = (V[1] - 2*V[0] + V[1]) / DX² = 2*(V[1] - V[0]) / DX²
        # Right boundary: d²V/dx²[N-1] = (V[N-2] - 2*V[N-1] + V[N-2]) / DX² = 2*(V[N-2] - V[N-1]) / DX²
        d2Vdx2 = d2Vdx2.at[0].set(2.0 * (V[1] - V[0]) / (DX**2))
        d2Vdx2 = d2Vdx2.at[N-1].set(2.0 * (V[N-2] - V[N-1]) / (DX**2))

        # Calculate ionic currents for all segments
        INa = gbar_Na * m**3 * h * (V - E_Na)
        IK  = gbar_K  * n**4 * (V - E_K)
        IL  = G_l * (V - E_l)

        # External current injection with smooth ramping (1ms ramp up/down)
        # Convert total current (μA) to current density (μA/cm²) based on segment size
        # This ensures same total current regardless of spatial discretization
        Iext = jnp.zeros(N)
        ramp_up = jnp.clip((t - T_START) / RAMP_T, 0.0, 1.0)
        ramp_down = jnp.clip((T_END - t) / RAMP_T, 0.0, 1.0)
        # I_amp is total current (μA), convert to density by dividing by segment membrane area
        segment_membrane_area = 2.0 * jnp.pi * A_RADIUS * DX  # cm²
        current_density = I_amp / segment_membrane_area  # μA/cm²
        current_value = current_density * ramp_up * ramp_down
        Iext = Iext.at[INJECTION_SEGMENT].set(current_value)

        # Cable equation: C_m * dV/dt = I_axial_density - I_ion + I_ext
        # I_axial = LAMBDA_CABLE * d²V/dx² where LAMBDA_CABLE = πa²/R_a in S*cm
        # Units: (S*cm) * (mV/cm²) = mV*S/cm = mA/cm (current per unit length)
        # Convert to current density (μA/cm²) by dividing by membrane area per unit length (2πa):
        # Current density = (mA/cm) / (2πa cm) * 1000 = μA/cm²
        membrane_circumference = 2.0 * jnp.pi * A_RADIUS  # cm
        spatial_current_density = (LAMBDA_CABLE * d2Vdx2 / membrane_circumference) * 1000.0  # μA/cm²
        dV = (spatial_current_density - INa - IK - IL + Iext) / C_m

        # Enforce reversal potential constraint: when V >= E_Na, dV/dt must be ≤ 0
        # This prevents V from exceeding E_Na regardless of injected current
        dV = jnp.where(V >= E_Na, jnp.minimum(dV, 0.0), dV)

        # Gating variable dynamics (same for all segments)
        dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n

        # Return concatenated derivatives
        return jnp.concatenate([dV, dm, dh, dn])

    # Initial conditions - tuple of (V_all, m_all, h_all, n_all) each of shape (N_SEGMENTS,)
    V0_all, m0_all, h0_all, n0_all = init_conditions

    # Clamp to valid ranges
    V0_all = jnp.clip(V0_all, -150.0, 100.0)
    m0_all = jnp.clip(m0_all, 0.0, 1.0)
    h0_all = jnp.clip(h0_all, 0.0, 1.0)
    n0_all = jnp.clip(n0_all, 0.0, 1.0)

    # Concatenate all initial conditions
    y0 = jnp.concatenate([V0_all, m0_all, h0_all, n0_all])

    # Setup ODE solver - use Tsit5 (5th order) with adaptive stepping for better efficiency
    term = diffrax.ODETerm(cable_hh_dynamics)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-6)

    # Save at 0.1 ms intervals instead of every timestep
    # This reduces memory by ~70× while maintaining sufficient resolution for analysis
    save_interval = 0.1  # ms
    ts = jnp.arange(0.0, tmax + save_interval/2, save_interval)
    saveat = diffrax.SaveAt(ts=ts)

    # Solve cable PDE with adaptive-step Tsit5 solver
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tmax,
        dt0=dt,  # Initial step size hint
        y0=y0,
        args=I_amplitude,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1000000,  # High limit for adaptive stepping (Tsit5 needs many steps)
        throw=False  # Don't throw errors, return result with error code
    )

    # Memory-efficient: Extract only the two voltage traces we need
    # Instead of returning full solution (800 vars × 70k timesteps = 560M values)
    # Return only 2 voltage traces (2 × 70k timesteps = 140k values per simulation)
    V_solution = solution.ys[:, :N_SEGMENTS]  # Shape: (n_times, N_SEGMENTS)

    # Enforce V ≤ E_Na constraint on the extracted segments
    V_injection = jnp.clip(V_solution[:, INJECTION_SEGMENT], -150.0, E_Na)
    V_distal = jnp.clip(V_solution[:, RECORDING_SEGMENT], -150.0, E_Na)

    return solution.ts, V_injection, V_distal


@partial(jit, static_argnames=('tmax', 'dt'))
def simulate_batch_jax(params_batch, tmax=100.0, dt=0.02, I_amplitude=10.0):
    """
    Simulate multiple parameter sets using pmap for multi-core parallelization.
    Uses spatially-extended cable model.
    Records voltage at injection site and distal site for each simulation.

    Strategy: Split batch across CPU cores using pmap, with lax.scan within each core.
    This balances parallelism with memory efficiency.

    Parameters:
    -----------
    params_batch : array
        Batch of parameter sets
    tmax : float
        Simulation duration (ms)
    dt : float
        Initial timestep hint (ms)
    I_amplitude : float
        Total current injection amplitude in μA

    Returns:
    --------
    times : array
        Time vector
    voltage_injection : array
        Voltage traces at injection site (n_sims, n_times)
    voltage_distal : array
        Voltage traces at distal site (n_sims, n_times)
    """
    # Vectorize equilibrium finding across batch
    batch_find_eq = vmap(find_equilibrium_jax)
    V_rest_batch, m0_batch, h0_batch, n0_batch = batch_find_eq(params_batch)

    # Stack initial conditions: shape (batch_size, 4, N_SEGMENTS)
    # We'll pass this through and unpack in simulate_HH_AP_jax
    init_conds_batch = jnp.stack([V_rest_batch, m0_batch, h0_batch, n0_batch], axis=1)

    # Get number of devices
    n_devices = len(jax.devices())
    batch_size = params_batch.shape[0]

    # Pad batch to be divisible by n_devices if needed
    remainder = batch_size % n_devices
    if remainder != 0:
        pad_size = n_devices - remainder
        params_batch = jnp.concatenate([
            params_batch,
            jnp.repeat(params_batch[-1:], pad_size, axis=0)
        ], axis=0)
        init_conds_batch = jnp.concatenate([
            init_conds_batch,
            jnp.repeat(init_conds_batch[-1:], pad_size, axis=0)
        ], axis=0)

    # Reshape to (n_devices, sims_per_device, ...)
    per_device = params_batch.shape[0] // n_devices
    params_split = params_batch.reshape(n_devices, per_device, -1)
    # For init_conds, preserve the (4, N_SEGMENTS) structure
    init_conds_split = init_conds_batch.reshape(n_devices, per_device, 4, N_SEGMENTS)

    # Define function to run on each device using lax.scan
    def process_device_batch(params_device, init_conds_device):
        def scan_fn(carry, inputs):
            params, init_conds = inputs
            # Unpack init_conds from stacked array to tuple
            init_tuple = (init_conds[0], init_conds[1], init_conds[2], init_conds[3])
            times, V_inj, V_dist = simulate_HH_AP_jax(params, init_tuple, tmax, dt, I_amplitude)
            return carry, (V_inj, V_dist)

        _, (V_inj_batch, V_dist_batch) = lax.scan(
            scan_fn,
            None,
            (params_device, init_conds_device)
        )
        return V_inj_batch, V_dist_batch

    # Use pmap to parallelize across devices
    pmap_fn = pmap(process_device_batch)
    voltage_injection_split, voltage_distal_split = pmap_fn(params_split, init_conds_split)

    # Reshape back to (total_batch_size, n_times)
    voltage_injection = voltage_injection_split.reshape(-1, voltage_injection_split.shape[-1])
    voltage_distal = voltage_distal_split.reshape(-1, voltage_distal_split.shape[-1])

    # Remove padding if we added any
    if remainder != 0:
        voltage_injection = voltage_injection[:batch_size]
        voltage_distal = voltage_distal[:batch_size]

    # Get times from first simulation (all identical)
    # Unpack first init_conds to tuple
    init_tuple_0 = (init_conds_batch[0, 0], init_conds_batch[0, 1], init_conds_batch[0, 2], init_conds_batch[0, 3])
    times, _, _ = simulate_HH_AP_jax(params_batch[0], init_tuple_0, tmax, dt, I_amplitude)

    return times, voltage_injection, voltage_distal


def incremental_simulation_jax(param_values, time_test, dt, nT, chunk_size=5000, save_path='temp_results_jax', I_amplitude=10.0):
    """
    Run spatially-extended cable model simulations in chunks using JAX vectorization.
    Records voltage at injection site and distal site for statistical analysis.
    Much faster than multiprocessing approach due to JIT compilation and vectorization.

    Parameters:
    -----------
    param_values : array
        Parameter sets from sampling (NumPy array)
    time_test : array
        Time vector
    dt : float
        Time step
    nT : int
        Number of time points
    chunk_size : int
        Number of simulations to run before saving
    save_path : str
        Directory to save temporary results
    I_amplitude : float
        Total current injection amplitude in μA

    Returns:
    --------
    save_path : str
        Path to directory containing chunk files
    num_chunks : int
        Number of chunk files created
    """
    import os
    import time
    import numpy as np

    # Create temporary directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_sims = len(param_values)

    # Warm-up compilation strategy:
    # - First chunk uses small size (50) to safely compile JIT code
    # - Subsequent chunks use larger size (500) to reuse compiled code for 10× speedup
    warmup_chunk_size = 50  # Safe for compilation
    production_chunk_size = chunk_size  # Use requested size after warm-up

    # Calculate chunks: first chunk is warmup_size, rest are production_size
    warmup_sims = min(warmup_chunk_size, total_sims)
    remaining_sims = total_sims - warmup_sims
    num_production_chunks = int(np.ceil(remaining_sims / production_chunk_size)) if remaining_sims > 0 else 0
    num_chunks = 1 + num_production_chunks  # 1 warmup + N production

    print(f"Running JAX simulations with warm-up compilation strategy:")
    print(f"  Chunk 0 (warm-up): {warmup_chunk_size} simulations - triggers JIT compilation")
    print(f"  Chunks 1-{num_chunks-1}: {production_chunk_size} simulations each - reuses compiled code")
    print(f"  Total chunks: {num_chunks}, Total simulations: {total_sims}")
    print(f"\nExpected performance:")
    print(f"  Chunk 0: ~30-50 seconds (compilation + execution)")
    print(f"  Chunks 1+: ~5-10 seconds each (execution only, 10× larger batches!)\n")

    # Pre-compute static values to avoid recompilation
    tmax_static = float(time_test[-1])
    dt_static = float(dt)

    for chunk_idx in range(num_chunks):
        # Determine chunk boundaries based on warm-up vs production
        if chunk_idx == 0:
            # Warm-up chunk: small size for safe compilation
            chunk_start = 0
            chunk_end = warmup_sims
            current_chunk_size = warmup_chunk_size
            chunk_type = "WARM-UP (JIT compilation)"
        else:
            # Production chunk: large size using compiled code
            prod_idx = chunk_idx - 1
            chunk_start = warmup_sims + prod_idx * production_chunk_size
            chunk_end = min(warmup_sims + (prod_idx + 1) * production_chunk_size, total_sims)
            current_chunk_size = production_chunk_size
            chunk_type = "PRODUCTION (reusing compiled code)"

        actual_chunk_size = chunk_end - chunk_start

        print(f"\nProcessing JAX chunk {chunk_idx + 1}/{num_chunks} - {chunk_type}")
        print(f"  Samples {chunk_start}-{chunk_end} ({actual_chunk_size} simulations)")
        chunk_start_time = time.time()

        # Get parameter subset for this chunk
        params_chunk = param_values[chunk_start:chunk_end]

        # Pad chunk to match expected size to avoid recompilation
        if actual_chunk_size < current_chunk_size:
            # Repeat last parameter set to fill chunk
            padding_size = current_chunk_size - actual_chunk_size
            padding = np.tile(params_chunk[-1:], (padding_size, 1))
            params_chunk = np.vstack([params_chunk, padding])
            print(f"  Note: Padded from {actual_chunk_size} to {current_chunk_size} samples")

        # Convert to JAX array
        params_jax = jnp.array(params_chunk)

        try:
            # Run vectorized simulation
            if chunk_idx == 0:
                print(f"  Compiling + running {current_chunk_size} simulations (I={I_amplitude} μA)...")
            else:
                print(f"  Running {current_chunk_size} simulations in parallel (I={I_amplitude} μA)...")
            sim_start = time.time()
            _, voltage_injection, voltage_distal = simulate_batch_jax(params_jax, tmax_static, dt_static, I_amplitude)
            sim_time = time.time() - sim_start

            # Trim padding from results if needed
            if actual_chunk_size < current_chunk_size:
                voltage_injection = voltage_injection[:actual_chunk_size]
                voltage_distal = voltage_distal[:actual_chunk_size]

            # Convert back to NumPy for saving
            chunk_V_injection = np.array(voltage_injection, dtype=np.float32)
            chunk_V_distal = np.array(voltage_distal, dtype=np.float32)

            print(f"  Simulations completed in {sim_time:.2f}s ({actual_chunk_size/sim_time:.1f} sims/sec)")

            # Save chunks to disk (separate files for each recording site)
            chunk_file_injection = os.path.join(save_path, f'chunk_{chunk_idx}_injection.npy')
            chunk_file_distal = os.path.join(save_path, f'chunk_{chunk_idx}_distal.npy')
            estimated_size_mb = (chunk_V_injection.nbytes + chunk_V_distal.nbytes) / (1024**2)
            print(f"  Saving chunk {chunk_idx + 1}/{num_chunks} (~{estimated_size_mb:.1f} MB)...")

            np.save(chunk_file_injection, chunk_V_injection)
            np.save(chunk_file_distal, chunk_V_distal)

            # Verify save
            if (os.path.exists(chunk_file_injection) and os.path.getsize(chunk_file_injection) > 1000 and
                os.path.exists(chunk_file_distal) and os.path.getsize(chunk_file_distal) > 1000):
                chunk_total_time = time.time() - chunk_start_time
                print(f"  ✓ Chunk {chunk_idx + 1}/{num_chunks} saved successfully (total: {chunk_total_time:.2f}s)")
            else:
                print(f"  WARNING: Chunk {chunk_idx + 1} may not have saved correctly")

        except Exception as e:
            print(f"  ERROR during chunk {chunk_idx + 1}: {e}")
            print(f"  Attempting to save partial results...")

            # Try saving smaller sub-batches
            try:
                sub_batch_size = min(1000, actual_chunk_size // 4)
                for sub_idx in range(0, actual_chunk_size, sub_batch_size):
                    sub_end = min(sub_idx + sub_batch_size, actual_chunk_size)
                    sub_params = params_jax[sub_idx:sub_end]
                    _, sub_injection, sub_distal = simulate_batch_jax(sub_params, tmax_static, dt_static, I_amplitude)
                    sub_chunk_inj = np.array(sub_injection, dtype=np.float32)
                    sub_chunk_dist = np.array(sub_distal, dtype=np.float32)
                    sub_file_inj = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx//sub_batch_size}_injection.npy')
                    sub_file_dist = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx//sub_batch_size}_distal.npy')
                    np.save(sub_file_inj, sub_chunk_inj)
                    np.save(sub_file_dist, sub_chunk_dist)
                print(f"  Chunk {chunk_idx + 1} saved as sub-chunks")
            except Exception as e2:
                print(f"  CRITICAL ERROR: Could not save chunk {chunk_idx + 1}: {e2}")
                raise

        # Free memory
        del chunk_V_injection, chunk_V_distal

    return save_path, num_chunks
