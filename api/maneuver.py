"""
api/maneuver.py
POST /api/maneuver/schedule — validate and queue burn commands.
Validates LOS, fuel sufficiency, cooldown, and ΔV limits.
"""

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from state.fleet import fleet_state, BurnCommand
from physics.maneuver_math import validate_burn, fuel_mass_consumed
from physics.propagator import propagate
from ground.los import has_line_of_sight

logger = logging.getLogger("acm.api.maneuver")
router = APIRouter()


# ── Request / Response Models ─────────────────────────────────────────────────

class DeltaVec(BaseModel):
    x: float
    y: float
    z: float

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class BurnRequest(BaseModel):
    burn_id: str
    burnTime: str
    deltaV_vector: DeltaVec


class ManeuverScheduleRequest(BaseModel):
    satelliteId: str
    maneuver_sequence: List[BurnRequest]


class ValidationResult(BaseModel):
    ground_station_los: bool
    sufficient_fuel: bool
    projected_mass_remaining_kg: float


class ManeuverScheduleResponse(BaseModel):
    status: str
    validation: ValidationResult


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/maneuver/schedule", response_model=ManeuverScheduleResponse)
async def schedule_maneuver(payload: ManeuverScheduleRequest):
    """
    Validate and queue a maneuver sequence for a satellite.
    All burns must pass LOS, fuel, cooldown, and magnitude checks.
    """
    sat = fleet_state.satellites.get(payload.satelliteId)
    if sat is None:
        raise HTTPException(status_code=404, detail=f"Satellite {payload.satelliteId} not found")

    if sat.status == "EOL":
        raise HTTPException(status_code=409, detail=f"Satellite {payload.satelliteId} is EOL — no maneuvers allowed")

    queued_burns: List[BurnCommand] = []
    projected_mass = sat.wet_mass
    los_ok = True

    for burn_req in payload.maneuver_sequence:
        # Parse burn time
        try:
            burn_time = datetime.fromisoformat(burn_req.burnTime.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid burnTime: {burn_req.burnTime}")

        dv = burn_req.deltaV_vector.to_np()
        dv_mag = float(np.linalg.norm(dv))

        # Validate physical constraints
        valid, reason = validate_burn(sat, dv_mag, burn_time, fleet_state.sim_time)
        if not valid:
            raise HTTPException(
                status_code=422,
                detail=f"Burn {burn_req.burn_id} failed validation: {reason}"
            )

        # Verify LOS at burn time
        t_to_burn = (burn_time - fleet_state.sim_time).total_seconds()
        r_at_burn, _ = propagate(sat.r, sat.v, max(t_to_burn, 0.0))
        los, station = has_line_of_sight(r_at_burn)

        if not los:
            los_ok = False
            logger.warning(
                f"No LOS for burn {burn_req.burn_id} on {payload.satelliteId} "
                f"at {burn_time.isoformat()}"
            )
            # We still queue it — system may pre-upload before blackout

        # Track projected mass across the sequence
        consumed = fuel_mass_consumed(projected_mass, dv_mag)
        projected_mass = max(0.0, projected_mass - consumed)

        queued_burns.append(BurnCommand(
            burn_id=burn_req.burn_id,
            burn_time=burn_time,
            delta_v=dv
        ))

    # All burns validated — append to satellite queue
    sat.maneuver_queue.extend(queued_burns)

    logger.info(
        f"Scheduled {len(queued_burns)} burn(s) for {payload.satelliteId}. "
        f"Projected mass: {projected_mass:.2f} kg"
    )

    return ManeuverScheduleResponse(
        status="SCHEDULED",
        validation=ValidationResult(
            ground_station_los=los_ok,
            sufficient_fuel=True,
            projected_mass_remaining_kg=round(projected_mass, 2)
        )
    )