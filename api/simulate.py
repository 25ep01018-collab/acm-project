"""
api/simulate.py
Every step:
- Checks debris proximity RIGHT NOW
- Fires burns immediately (no future scheduling nonsense)
- Updates fuel, position, status instantly
- Dashboard shows changes on next poll
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from fastapi import APIRouter
from pydantic import BaseModel
from state.fleet import fleet_state, BurnCommand
from physics.propagator import propagate

logger = logging.getLogger("acm.simulate")
router = APIRouter()

COLLISION_KM   = 0.100   # 100m = collision
WARNING_KM     = 5.0     # 5km  = warning
EVASION_DV     = 0.012   # 12 m/s evasion burn (km/s)
RECOVERY_DV    = 0.012   # 12 m/s recovery burn
COOLDOWN_S     = 600.0   # 10 min between burns
DRY_MASS       = 500.0
ISP            = 300.0
G0             = 0.0098066  # km/s²

class SimStepRequest(BaseModel):
    step_seconds: int

class SimStepResponse(BaseModel):
    status: str
    new_timestamp: str
    collisions_detected: int
    maneuvers_executed: int


def tsiolkovsky(m_wet, dv_km_s):
    """Fuel consumed for a burn."""
    import math
    return m_wet * (1 - math.exp(-abs(dv_km_s) / (ISP * G0)))


def can_burn(sat, sim_time):
    """Check cooldown."""
    if sat.last_burn_time is None:
        return True
    gap = (sim_time - sat.last_burn_time).total_seconds()
    return gap >= COOLDOWN_S


def fire_burn(sat, dv_vec, burn_id, sim_time):
    """
    Instantly apply a burn to the satellite.
    Updates velocity, fuel, status immediately.
    """
    dv_mag = float(np.linalg.norm(dv_vec))
    if dv_mag < 1e-9:
        return False

    m_wet   = sat.dry_mass + sat.fuel_kg
    delta_m = tsiolkovsky(m_wet, dv_mag)

    if delta_m >= sat.fuel_kg:
        logger.warning(f"{sat.sat_id} not enough fuel for burn")
        return False

    # Apply burn
    sat.v           = sat.v + dv_vec
    sat.fuel_kg     = max(0.0, sat.fuel_kg - delta_m)
    sat.last_burn_time = sim_time
    sat.total_dv_used += dv_mag

    logger.info(
        f"BURN: {sat.sat_id} | {burn_id} | "
        f"dv={dv_mag*1000:.2f} m/s | "
        f"fuel={sat.fuel_kg:.3f} kg remaining"
    )
    return True


@router.post("/simulate/step", response_model=SimStepResponse)
async def simulate_step(payload: SimStepRequest):
    dt      = float(payload.step_seconds)
    t_now   = fleet_state.sim_time
    t_end   = t_now + timedelta(seconds=dt)

    collisions = 0
    burns_fired = 0

    # ── Build debris KD-tree ──────────────────────────────────────────────────
    debris_list = list(fleet_state.debris.values())
    if debris_list:
        from scipy.spatial import KDTree
        dpos = np.array([d.r for d in debris_list])
        tree = KDTree(dpos)
    else:
        tree = None

    # ── For each satellite: check proximity, fire burns, propagate ────────────
    for sat in fleet_state.satellites.values():

        if sat.status == "EOL":
            sat.r, sat.v = propagate(sat.r, sat.v, dt)
            if sat.nominal_slot_r is not None:
                sat.nominal_slot_r, sat.nominal_slot_v = propagate(
                    sat.nominal_slot_r, sat.nominal_slot_v, dt)
            continue

        if tree is not None:
            # Find closest debris right now
            nearby = tree.query_ball_point(sat.r, WARNING_KM)

            if nearby:
                # Get the single closest debris
                dists = [np.linalg.norm(sat.r - debris_list[i].r) for i in nearby]
                closest_idx = nearby[np.argmin(dists)]
                closest_deb = debris_list[closest_idx]
                miss_km     = float(np.linalg.norm(sat.r - closest_deb.r))

                # COLLISION detected
                if miss_km < COLLISION_KM:
                    collisions += 1
                    logger.error(f"COLLISION: {sat.sat_id} hit {closest_deb.deb_id} at {miss_km*1000:.0f}m")

                # EVASION: debris within warning range and cooldown ok
                if miss_km < WARNING_KM and can_burn(sat, t_now):
                    # Direction AWAY from debris
                    away = sat.r - closest_deb.r
                    away_norm = away / np.linalg.norm(away)
                    dv_vec = away_norm * EVASION_DV

                    ok = fire_burn(sat, dv_vec, f"EVASION_{closest_deb.deb_id}", t_now)
                    if ok:
                        sat.status = "EVADING"
                        burns_fired += 1

                        # Schedule recovery burn info (for logging)
                        logger.info(
                            f"EVASION fired: {sat.sat_id} away from {closest_deb.deb_id} "
                            f"(miss was {miss_km*1000:.0f}m)"
                        )

            else:
                # No debris nearby — if evading, start recovering
                if sat.status == "EVADING":
                    sat.status = "RECOVERING"

                # Recovery: push back toward nominal slot
                if sat.status == "RECOVERING" and can_burn(sat, t_now):
                    if sat.nominal_slot_r is not None:
                        slot_dir = sat.nominal_slot_r - sat.r
                        slot_dist = float(np.linalg.norm(slot_dir))

                        if slot_dist > 10.0:  # still outside 10km box
                            dv_vec = (slot_dir / slot_dist) * min(RECOVERY_DV, slot_dist * 0.001)
                            ok = fire_burn(sat, dv_vec, "RECOVERY", t_now)
                            if ok:
                                burns_fired += 1
                        else:
                            sat.status = "NOMINAL"
                            logger.info(f"{sat.sat_id} recovered to nominal slot")

        # EOL check
        if sat.fuel_kg / 50.0 <= 0.05 and sat.status != "EOL":
            sat.status = "EOL"
            logger.warning(f"EOL: {sat.sat_id} fuel={sat.fuel_kg:.2f}kg — moving to graveyard")
            # Final burn: prograde to raise orbit
            if can_burn(sat, t_now):
                r_hat = sat.r / np.linalg.norm(sat.r)
                v_hat = sat.v / np.linalg.norm(sat.v)
                dv_vec = v_hat * min(0.010, sat.fuel_kg * ISP * G0 / (sat.dry_mass + sat.fuel_kg))
                fire_burn(sat, dv_vec, "EOL_GRAVEYARD", t_now)

        # Propagate satellite
        sat.r, sat.v = propagate(sat.r, sat.v, dt)
        if sat.nominal_slot_r is not None:
            sat.nominal_slot_r, sat.nominal_slot_v = propagate(
                sat.nominal_slot_r, sat.nominal_slot_v, dt)

        # Station keeping outage tracking
        if not sat.in_station_box():
            sat.service_outage_seconds += dt

    # ── Propagate all debris ──────────────────────────────────────────────────
    for deb in fleet_state.debris.values():
        deb.r, deb.v = propagate(deb.r, deb.v, dt)

    # ── Advance clock ─────────────────────────────────────────────────────────
    fleet_state.sim_time = t_end

    # ── Update CDM warnings for dashboard display ─────────────────────────────
    if debris_list and tree is not None:
        from state.fleet import CDMWarning
        warnings = []
        for sat in fleet_state.satellites.values():
            nearby = tree.query_ball_point(sat.r, WARNING_KM)
            for idx in nearby:
                deb  = debris_list[idx]
                miss = float(np.linalg.norm(sat.r - deb.r))
                rel_v = float(np.linalg.norm(sat.v - deb.v))
                sev  = "CRITICAL" if miss < COLLISION_KM else "WARNING"
                warnings.append(CDMWarning(
                    sat_id=sat.sat_id,
                    deb_id=deb.deb_id,
                    tca_time=t_end,
                    miss_distance_km=round(miss, 6),
                    relative_velocity_km_s=round(rel_v, 4),
                    severity=sev
                ))
        fleet_state.cdm_warnings = sorted(
            warnings, key=lambda w: w.miss_distance_km)[:100]

    return SimStepResponse(
        status="STEP_COMPLETE",
        new_timestamp=t_end.isoformat().replace("+00:00", "Z"),
        collisions_detected=collisions,
        maneuvers_executed=burns_fired
    )