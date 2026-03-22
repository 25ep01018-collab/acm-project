"""
physics/maneuver_math.py
All maneuver planning mathematics:
  - RTN ↔ ECI frame conversion
  - Tsiolkovsky rocket equation fuel calculations
  - Evasion burn planning (prograde/radial shunt)
  - Recovery/station-keeping burn planning (Hohmann / phasing)
  - EOL graveyard orbit maneuver
"""

import math
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional

from physics.propagator import propagate
from state.fleet import (
    SatelliteObject, BurnCommand,
    MU, ISP_S, G0_KM_S2, MAX_DV_PER_BURN,
    THRUSTER_COOLDOWN, DRY_MASS_KG
)

logger = logging.getLogger("acm.maneuver")

# Graveyard orbit altitude (km above Earth surface)
GRAVEYARD_ALTITUDE_KM = 36000.0
RE = 6378.137


# ── RTN Frame ─────────────────────────────────────────────────────────────────

def compute_rtn_frame(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute RTN rotation matrix (rows = R̂, T̂, N̂ unit vectors in ECI).
    r: ECI position (km)
    v: ECI velocity (km/s)
    Returns 3×3 rotation matrix: ECI_vec = M.T @ RTN_vec
    """
    r_hat = r / np.linalg.norm(r)                    # Radial
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)                    # Normal
    t_hat = np.cross(n_hat, r_hat)                   # Transverse

    # Rows are RTN unit vectors — multiply by RTN vector to get ECI
    return np.array([r_hat, t_hat, n_hat])


def rtn_to_eci(dv_rtn: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert a delta-V vector from RTN frame to ECI frame.
    dv_rtn: [dv_R, dv_T, dv_N] in km/s
    Returns dv in ECI km/s
    """
    M = compute_rtn_frame(r, v)
    return M.T @ dv_rtn


def eci_to_rtn(dv_eci: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert a delta-V vector from ECI frame to RTN frame.
    """
    M = compute_rtn_frame(r, v)
    return M @ dv_eci


# ── Tsiolkovsky Rocket Equation ───────────────────────────────────────────────

def fuel_mass_consumed(m_current_kg: float, dv_km_s: float) -> float:
    """
    Propellant mass consumed for a given ΔV.
    Uses: Δm = m_current * (1 - exp(-|ΔV| / (Isp * g0)))
    Returns mass in kg.
    """
    return m_current_kg * (1.0 - math.exp(-abs(dv_km_s) / (ISP_S * G0_KM_S2)))


def max_dv_available(m_wet_kg: float) -> float:
    """
    Maximum ΔV achievable given current wet mass (Tsiolkovsky).
    Returns ΔV in km/s.
    """
    return ISP_S * G0_KM_S2 * math.log(m_wet_kg / DRY_MASS_KG)


def validate_burn(sat: SatelliteObject, dv_mag_km_s: float,
                  burn_time: datetime, sim_time: datetime) -> Tuple[bool, str]:
    """
    Validate that a burn command is physically feasible.
    Returns (valid: bool, reason: str)
    """
    # Magnitude limit
    if dv_mag_km_s > MAX_DV_PER_BURN:
        return False, f"ΔV {dv_mag_km_s*1000:.1f} m/s exceeds limit of {MAX_DV_PER_BURN*1000:.0f} m/s"

    # Fuel check: ensure we have enough
    fuel_needed = fuel_mass_consumed(sat.wet_mass, dv_mag_km_s)
    if fuel_needed >= sat.fuel_kg:
        return False, f"Insufficient fuel: need {fuel_needed:.2f} kg, have {sat.fuel_kg:.2f} kg"

    # Cooldown check
    if sat.last_burn_time is not None:
        gap = (burn_time - sat.last_burn_time).total_seconds()
        if gap < THRUSTER_COOLDOWN:
            return False, f"Thruster cooldown: only {gap:.0f}s since last burn (need {THRUSTER_COOLDOWN}s)"

    # Must be at least 10s in the future (signal latency)
    if (burn_time - sim_time).total_seconds() < 10:
        return False, "Burn time must be ≥10s in the future (signal latency)"

    return True, "OK"


# ── Evasion Burn Planning ─────────────────────────────────────────────────────

def plan_evasion_burn(
        sat: SatelliteObject,
        deb_r: np.ndarray,
        deb_v: np.ndarray,
        tca_seconds_from_now: float,
        sim_time: datetime
) -> Optional[BurnCommand]:
    """
    Plan an evasion burn to avoid a predicted conjunction.

    Strategy: apply a radial shunt (R component) perpendicular to the
    relative velocity vector. This is fuel-efficient and avoids in-plane
    orbit adjustments that are expensive to recover from.

    The burn is scheduled as early as possible (max lead time = better deflection).
    """
    # Schedule burn at latest possible pre-TCA time with LOS margin
    # Burn at TCA - 30 min minimum, or now + 60s if TCA is close
    lead_time = min(tca_seconds_from_now * 0.5, 1800.0)  # 30 min max lead
    lead_time = max(lead_time, 60.0)

    burn_offset_s = tca_seconds_from_now - lead_time
    burn_time = sim_time + timedelta(seconds=max(burn_offset_s, 60.0))

    # Propagate satellite to burn time
    r_burn, v_burn = propagate(sat.r, sat.v, max(burn_offset_s, 60.0))
    r_deb_at_burn, _ = propagate(deb_r, deb_v, max(burn_offset_s, 60.0))

    # Relative position at burn time
    rel_r = r_deb_at_burn - r_burn

    # Evasion direction: radial shunt away from debris
    # If debris is "above" (positive R), push down; if "below", push up
    M = compute_rtn_frame(r_burn, v_burn)
    rel_r_rtn = M @ rel_r  # debris position in RTN

    # Choose radial or transverse component for evasion
    if abs(rel_r_rtn[0]) > abs(rel_r_rtn[1]):
        # Radial shunt — opposite sign to debris radial position
        dv_rtn = np.array([-np.sign(rel_r_rtn[0]) * 0.010, 0.0, 0.0])
    else:
        # Prograde/retrograde shunt — phase change
        dv_rtn = np.array([0.0, -np.sign(rel_r_rtn[1]) * 0.008, 0.0])

    # Ensure we stay within per-burn limit
    dv_mag = np.linalg.norm(dv_rtn)
    if dv_mag > MAX_DV_PER_BURN:
        dv_rtn = dv_rtn * (MAX_DV_PER_BURN / dv_mag)

    # Convert to ECI
    dv_eci = rtn_to_eci(dv_rtn, r_burn, v_burn)

    valid, reason = validate_burn(sat, float(np.linalg.norm(dv_eci)), burn_time, sim_time)
    if not valid:
        logger.warning(f"Evasion burn invalid for {sat.sat_id}: {reason}")
        return None

    burn_id = f"EVASION_{sat.sat_id}_{burn_time.strftime('%H%M%S')}"
    return BurnCommand(
        burn_id=burn_id,
        burn_time=burn_time,
        delta_v=dv_eci
    )


# ── Recovery Burn Planning ────────────────────────────────────────────────────

def plan_recovery_burn(
        sat: SatelliteObject,
        evasion_burn_time: datetime,
        sim_time: datetime
) -> Optional[BurnCommand]:
    """
    Plan a recovery burn to return satellite to its nominal station-keeping box.

    Strategy: schedule 90 minutes after evasion (debris has passed), then
    compute a corrective prograde/retrograde burn to adjust orbital period
    and phase the satellite back to its nominal slot.
    """
    if sat.nominal_slot_r is None:
        return None

    # Wait for debris to pass: schedule 90 min after evasion burn
    recovery_offset_s = (evasion_burn_time - sim_time).total_seconds() + 5400.0
    recovery_time = sim_time + timedelta(seconds=recovery_offset_s)

    # Propagate satellite to recovery time
    r_rec, v_rec = propagate(sat.r, sat.v, recovery_offset_s)
    r_nom, v_nom = propagate(sat.nominal_slot_r, sat.nominal_slot_v, recovery_offset_s)

    # Compute velocity difference needed to match nominal orbit
    dv_eci = v_nom - v_rec

    # Cap at max burn limit
    dv_mag = float(np.linalg.norm(dv_eci))
    if dv_mag > MAX_DV_PER_BURN:
        dv_eci = dv_eci * (MAX_DV_PER_BURN / dv_mag)
        dv_mag = MAX_DV_PER_BURN

    if dv_mag < 1e-6:
        logger.info(f"{sat.sat_id}: already near nominal orbit, no recovery needed")
        return None

    valid, reason = validate_burn(sat, dv_mag, recovery_time, sim_time)
    if not valid:
        logger.warning(f"Recovery burn invalid for {sat.sat_id}: {reason}")
        return None

    burn_id = f"RECOVERY_{sat.sat_id}_{recovery_time.strftime('%H%M%S')}"
    return BurnCommand(
        burn_id=burn_id,
        burn_time=recovery_time,
        delta_v=dv_eci
    )


# ── EOL Graveyard Orbit ───────────────────────────────────────────────────────

def plan_eol_graveyard_burn(
        sat: SatelliteObject,
        sim_time: datetime
) -> Optional[BurnCommand]:
    """
    Plan a final raise-perigee burn to move a fuel-critical satellite
    into a graveyard orbit above operational LEO.
    Uses a prograde Hohmann transfer burn.
    """
    r_mag = float(np.linalg.norm(sat.r))
    v_mag = float(np.linalg.norm(sat.v))

    # Target semi-major axis for graveyard orbit
    a_graveyard = RE + GRAVEYARD_ALTITUDE_KM

    # Hohmann transfer: first burn at current orbit (circular approx)
    a_current = r_mag  # approximate as circular
    a_transfer = (a_current + a_graveyard) / 2.0

    # ΔV for Hohmann transfer departure burn
    v_current  = math.sqrt(MU / a_current)
    v_transfer = math.sqrt(MU * (2.0/a_current - 1.0/a_transfer))
    dv_needed  = v_transfer - v_current

    # Cap to remaining fuel capability
    dv_available = max_dv_available(sat.wet_mass)
    dv_burn = min(abs(dv_needed), min(dv_available * 0.9, MAX_DV_PER_BURN))

    # Apply prograde (transverse direction)
    dv_rtn = np.array([0.0, dv_burn, 0.0])
    dv_eci = rtn_to_eci(dv_rtn, sat.r, sat.v)

    burn_time = sim_time + timedelta(seconds=60.0)
    burn_id = f"EOL_GRAVEYARD_{sat.sat_id}"

    logger.warning(
        f"EOL burn scheduled for {sat.sat_id}: "
        f"ΔV={dv_burn*1000:.1f} m/s prograde (fuel={sat.fuel_kg:.2f} kg)"
    )

    return BurnCommand(
        burn_id=burn_id,
        burn_time=burn_time,
        delta_v=dv_eci
    )


# ── Apply Burn ────────────────────────────────────────────────────────────────

def apply_burn(sat: SatelliteObject, burn: BurnCommand) -> dict:
    """
    Apply an impulsive burn to a satellite's velocity vector.
    Deducts fuel using Tsiolkovsky equation.
    Returns a dict with execution summary.
    """
    dv_mag = float(np.linalg.norm(burn.delta_v))
    delta_m = fuel_mass_consumed(sat.wet_mass, dv_mag)

    # Impulsive: velocity changes instantly, position unchanged
    sat.v = sat.v + burn.delta_v
    sat.fuel_kg = max(0.0, sat.fuel_kg - delta_m)
    sat.last_burn_time = burn.burn_time
    sat.total_dv_used += dv_mag
    burn.executed = True

    # Update status
    if sat.is_eol:
        sat.status = "EOL"
    elif sat.status == "EVADING":
        sat.status = "RECOVERING"

    logger.info(
        f"Burn executed: {burn.burn_id} on {sat.sat_id} | "
        f"ΔV={dv_mag*1000:.2f} m/s | "
        f"Δm={delta_m:.3f} kg | "
        f"fuel_remaining={sat.fuel_kg:.2f} kg"
    )

    return {
        "burn_id": burn.burn_id,
        "dv_magnitude_ms": round(dv_mag * 1000, 3),
        "mass_consumed_kg": round(delta_m, 4),
        "fuel_remaining_kg": round(sat.fuel_kg, 3),
    }