# sim_core.py
"""
JAX/diffrax simulation engine for Hodgkin-Huxley model.
Contains all JAX-compiled functions and simulation logic.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import diffrax
from functools import partial
from typing import Tuple, Any
import config

# JAX configuration
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


@jit
def find_equilibrium_jax(params: jnp.ndarray) -> Tuple[float, float, float, float]:
    """
    Solve for an equilibrium (resting) potential V_rest by forcing net ionic current = 0
    with gating variables at their steady-state values.
    Uses Newton's method implemented in JAX for JIT compilation.

    Parameters
    ----------
    params : jnp.ndarray
        Model parameters

    Returns
    -------
    V_rest, m0, h0, n0 : float
        Resting potential and initial gating variables
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
        return A_bm * jnp.exp(-V / D_bm)

    def alpha_h(V):
        return A_ah * jnp.exp(-V / D_ah)

    def beta_h(V):
        return E_bh / (1.0 + jnp.exp(-(V + F_bh) / G_bh))

    def alpha_n(V):
        denom = 1.0 - jnp.exp(-(V_alpha + V) / k_alpha)
        return jnp.where(jnp.abs(denom) > 1e-9, A_alpha * (V_alpha + V) / denom, A_alpha * k_alpha)

    def beta_n(V):
        return A_beta * jnp.exp(-V / tau_beta)

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

    for i in range(50):  # More iterations with damping
        f = net_current(V)

        # Numerical derivative with small step
        h_step = 1e-7
        df = (net_current(V + h_step) - f) / h_step

        # Prevent division by zero or very small derivatives
        df = jnp.where(jnp.abs(df) < 1e-10, 1e-10, df)

        # Newton step with damping
        dV = -f / df
        # Limit step size to prevent overshooting
        dV = jnp.clip(dV, -10.0, 10.0)

        V_new = V + 0.5 * dV  # Damping factor of 0.5

        # Enforce bounds
        V_new = jnp.clip(V_new, V_min, V_max)

        # Check for convergence
        change = jnp.abs(V_new - V)
        V = V_new

        # Early exit if converged (but keep iterating for JIT)
        V = jnp.where(change < 1e-6, V, V)

    V_rest = V
    m0 = m_inf(V_rest)
    h0 = h_inf(V_rest)
    n0 = n_inf(V_rest)

    return V_rest, m0, h0, n0


@partial(jit, static_argnames=('tmax', 'dt'))
def simulate_HH_AP_jax(params: jnp.ndarray, init_conditions: jnp.ndarray,
                      tmax: float = config.TMAX, dt: float = config.DT,
                      I_amplitude: float = config.I_AMPLITUDE_DEFAULT) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate the Hodgkin-Huxley action potential using diffrax ODE solver.
    JAX-compatible and JIT-compiled for performance.

    Parameters
    ----------
    params : jnp.ndarray
        Model parameters
    init_conditions : jnp.ndarray
        Initial conditions [V0, m0, h0, n0]
    tmax : float
        Maximum simulation time
    dt : float
        Time step
    I_amplitude : float
        Current injection amplitude in μA (converted to μA/cm² internally)

    Returns
    -------
    times, solution : jnp.ndarray
        Time vector and voltage trace solution
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
        return A_bm * jnp.exp(-V / D_bm)

    def alpha_h(V):
        return A_ah * jnp.exp(-V / D_ah)

    def beta_h(V):
        return E_bh / (1.0 + jnp.exp(-(V + F_bh)/G_bh))

    # K gating
    def alpha_n(V):
        denom = 1.0 - jnp.exp(-(V_alpha + V) / k_alpha)
        return jnp.where(jnp.abs(denom) > 1e-9, A_alpha*(V_alpha + V)/denom, A_alpha*k_alpha)

    def beta_n(V):
        return A_beta * jnp.exp(-V / tau_beta)

    # Injected current function (parameterized amplitude from args)
    def I_inj(t, I_amp):
        # Convert from μA to μA/cm² using standard membrane area
        current_density = I_amp / config.MEMBRANE_AREA_CM2
        return jnp.where((t >= config.STIMULUS_START) & (t < config.STIMULUS_END), current_density, 0.0)

    # ODE system for diffrax with safeguards
    def hh_dynamics(t, y, args):
        I_amp = args
        V, m, h, n = y

        # Clamp state variables to valid ranges to prevent numerical issues
        V = jnp.clip(V, -150.0, 100.0)
        m = jnp.clip(m, 0.0, 1.0)
        h = jnp.clip(h, 0.0, 1.0)
        n = jnp.clip(n, 0.0, 1.0)

        INa = gbar_Na * m**3 * h * (V - E_Na)
        IK  = gbar_K  * n**4       * (V - E_K)
        IL  = G_l * (V - E_l)
        Iext = I_inj(t, I_amp)

        dV = (-INa - IK - IL + Iext) / C_m
        dm = alpha_m(V)*(1.0 - m) - beta_m(V)*m
        dh = alpha_h(V)*(1.0 - h) - beta_h(V)*h
        dn = alpha_n(V)*(1.0 - n) - beta_n(V)*n

        return jnp.array([dV, dm, dh, dn])

    # Initial conditions - clamp to valid ranges
    y0 = jnp.array(init_conditions)
    V0, m0, h0, n0 = y0
    y0 = jnp.array([
        jnp.clip(V0, -150.0, 100.0),
        jnp.clip(m0, 0.0, 1.0),
        jnp.clip(h0, 0.0, 1.0),
        jnp.clip(n0, 0.0, 1.0)
    ])

    # Setup ODE solver with fixed stepping (faster, more predictable)
    term = diffrax.ODETerm(hh_dynamics)
    solver = diffrax.Tsit5()  # 5th order Runge-Kutta
    # Use fixed timestep for speed - dt=0.005 is small enough for HH model
    stepsize_controller = diffrax.ConstantStepSize()
    saveat = diffrax.SaveAt(ts=jnp.arange(0, tmax+dt, dt))

    # Solve ODE with fixed step size (much faster than adaptive)
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=tmax,
        dt0=dt,
        y0=y0,
        args=I_amplitude,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(tmax/dt) + 100,  # Just enough steps for fixed dt
        throw=False  # Don't throw errors, return result with error code
    )

    return solution.ts, solution.ys


# Vectorized simulation using vmap
@partial(jit, static_argnames=('tmax', 'dt'))
def simulate_batch_jax(params_batch: jnp.ndarray,
                      tmax: float = config.TMAX, dt: float = config.DT,
                      I_amplitude: float = config.I_AMPLITUDE_DEFAULT) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate multiple parameter sets in parallel using JAX vmap.
    This is the key function for massive speedup.

    Parameters
    ----------
    params_batch : jnp.ndarray
        Batch of parameter sets (shape: batch_size x num_params)
    tmax : float
        Maximum simulation time
    dt : float
        Time step
    I_amplitude : float
        Current injection amplitude in μA (converted to μA/cm² internally)

    Returns
    -------
    times, voltage_traces : jnp.ndarray
        Time vector and voltage traces (shape: batch_size x time_steps)
    """
    # Vectorize equilibrium finding across batch
    batch_find_eq = vmap(find_equilibrium_jax)
    V_rest_batch, m0_batch, h0_batch, n0_batch = batch_find_eq(params_batch)

    # Stack initial conditions
    init_conds_batch = jnp.stack([V_rest_batch, m0_batch, h0_batch, n0_batch], axis=1)

    # Vectorize simulation across batch
    batch_simulate = vmap(lambda p, ic: simulate_HH_AP_jax(p, ic, tmax, dt, I_amplitude))
    times, solutions = batch_simulate(params_batch, init_conds_batch)

    # Extract voltage traces (first column of solutions)
    # solutions shape: (batch_size, time_steps, 4)
    voltage_traces = solutions[:, :, 0]

    return times[0], voltage_traces