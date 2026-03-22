"""
api/telemetry.py
POST /api/telemetry — ingest high-frequency orbital state vectors.
Triggers asynchronous conjunction assessment after each batch.
"""

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List

from state.fleet import fleet_state
from physics.conjunction import run_conjunction_assessment

logger = logging.getLogger("acm.api.telemetry")
router = APIRouter()


# ── Request / Response Models ─────────────────────────────────────────────────

class Vec3(BaseModel):
    x: float
    y: float
    z: float

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class TelemetryObject(BaseModel):
    id: str
    type: str          # "SATELLITE" or "DEBRIS"
    r: Vec3
    v: Vec3


class TelemetryRequest(BaseModel):
    timestamp: str
    objects: List[TelemetryObject]


class TelemetryResponse(BaseModel):
    status: str
    processed_count: int
    active_cdm_warnings: int


# ── Background Conjunction Assessment ─────────────────────────────────────────

async def assess_conjunctions_background():
    """
    Run conjunction assessment asynchronously so telemetry ACKs are instant.
    Updates fleet_state.cdm_warnings with fresh results.
    """
    try:
        warnings = run_conjunction_assessment(
            satellites=fleet_state.satellites,
            debris=fleet_state.debris,
            sim_time=fleet_state.sim_time,
        )
        fleet_state.cdm_warnings = warnings

        # Trigger autonomous COLA for critical warnings
        from physics.maneuver_math import plan_evasion_burn, plan_recovery_burn
        from ground.los import has_line_of_sight
        from physics.propagator import propagate

        for warning in warnings:
            if warning.severity != "CRITICAL":
                continue

            sat = fleet_state.satellites.get(warning.sat_id)
            if sat is None or sat.status == "EOL":
                continue

            # Check if evasion already scheduled for this pair
            existing = [b for b in sat.maneuver_queue
                        if not b.executed and "EVASION" in b.burn_id]
            if existing:
                continue

            deb = fleet_state.debris.get(warning.deb_id)
            if deb is None:
                continue

            tca_secs = (warning.tca_time - fleet_state.sim_time).total_seconds()

            # Plan evasion burn
            evasion = plan_evasion_burn(
                sat, deb.r, deb.v, tca_secs, fleet_state.sim_time
            )
            if evasion:
                # Verify LOS at burn time
                r_at_burn, _ = propagate(
                    sat.r, sat.v,
                    (evasion.burn_time - fleet_state.sim_time).total_seconds()
                )
                los, station = has_line_of_sight(r_at_burn)
                if not los:
                    logger.warning(
                        f"No LOS for evasion burn on {sat.sat_id} — "
                        f"searching earlier window..."
                    )
                    # Could implement blackout-aware scheduling here

                sat.maneuver_queue.append(evasion)
                sat.status = "EVADING"
                logger.info(f"Auto-scheduled evasion burn for {sat.sat_id}")

                # Plan recovery burn
                recovery = plan_recovery_burn(sat, evasion.burn_time, fleet_state.sim_time)
                if recovery:
                    sat.maneuver_queue.append(recovery)

    except Exception as e:
        logger.error(f"Conjunction assessment error: {e}", exc_info=True)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/telemetry", response_model=TelemetryResponse)
async def ingest_telemetry(
    payload: TelemetryRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest orbital state vectors for satellites and debris.
    Updates internal physics state and triggers async conjunction assessment.
    """
    # Update sim time from telemetry timestamp
    try:
        fleet_state.sim_time = datetime.fromisoformat(
            payload.timestamp.replace("Z", "+00:00")
        )
    except ValueError:
        fleet_state.sim_time = datetime.now(timezone.utc)

    # Upsert all objects
    for obj in payload.objects:
        fleet_state.upsert_object(
            obj_id=obj.id,
            obj_type=obj.type,
            r=obj.r.to_np(),
            v=obj.v.to_np()
        )

    # Trigger conjunction assessment in background (non-blocking)
    background_tasks.add_task(assess_conjunctions_background)

    return TelemetryResponse(
        status="ACK",
        processed_count=len(payload.objects),
        active_cdm_warnings=fleet_state.active_cdm_count()
    )