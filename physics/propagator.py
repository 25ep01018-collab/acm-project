"""
physics/propagator.py
RK4 numerical integrator with J2 gravitational perturbation.
Propagates 6D ECI state vectors [x,y,z,vx,vy,vz] forward in time.
"""

import numpy as np
from state.fleet import MU, RE, J2


def j2_acceleration(r: np.ndarray) -> np.ndarray:
    """
    Compute J2 perturbation acceleration vector in ECI frame.
    r: position vector [x, y, z] in km
    Returns acceleration in km/s²
    """
    x, y, z = r
    r_mag = np.linalg.norm(r)
    r2 = r_mag ** 2
    z2 = z ** 2

    factor = (3.0 / 2.0) * J2 * MU * RE**2 / (r_mag ** 5)

    ax = factor * x * (5.0 * z2 / r2 - 1.0)
    ay = factor * y * (5.0 * z2 / r2 - 1.0)
    az = factor * z * (5.0 * z2 / r2 - 3.0)

    return np.array([ax, ay, az])


def equations_of_motion(state: np.ndarray) -> np.ndarray:
    """
    Equations of motion including J2.
    state: [x, y, z, vx, vy, vz]
    Returns: [vx, vy, vz, ax, ay, az]
    """
    r = state[:3]
    v = state[3:]
    r_mag = np.linalg.norm(r)

    # Two-body gravity
    a_grav = -(MU / r_mag**3) * r

    # J2 perturbation
    a_j2 = j2_acceleration(r)

    a_total = a_grav + a_j2

    return np.concatenate([v, a_total])


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """
    Single RK4 integration step.
    state: [x, y, z, vx, vy, vz]
    dt:    time step in seconds
    Returns new state after dt seconds.
    """
    k1 = equations_of_motion(state)
    k2 = equations_of_motion(state + 0.5 * dt * k1)
    k3 = equations_of_motion(state + 0.5 * dt * k2)
    k4 = equations_of_motion(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate(r: np.ndarray, v: np.ndarray,
              total_seconds: float, dt: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate a state vector forward by total_seconds using RK4 with adaptive sub-steps.
    r:             position ECI (km)
    v:             velocity ECI (km/s)
    total_seconds: total propagation time (s)
    dt:            RK4 sub-step size (s), default 30s for accuracy
    Returns: (new_r, new_v)
    """
    state = np.concatenate([r, v])
    remaining = total_seconds
    while remaining > 0:
        step = min(dt, remaining)
        state = rk4_step(state, step)
        remaining -= step
    return state[:3], state[3:]


def propagate_states(r: np.ndarray, v: np.ndarray,
                     times: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Propagate a state to multiple future times (sorted ascending from t=0).
    Useful for TCA search.
    Returns list of (r, v) tuples at each time.
    """
    results = []
    state = np.concatenate([r, v])
    prev_t = 0.0
    for t in times:
        dt_step = t - prev_t
        if dt_step > 0:
            remaining = dt_step
            while remaining > 0:
                step = min(30.0, remaining)
                state = rk4_step(state, step)
                remaining -= step
        results.append((state[:3].copy(), state[3:].copy()))
        prev_t = t
    return results