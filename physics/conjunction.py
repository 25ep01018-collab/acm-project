"""
physics/conjunction.py
Predictive Conjunction Assessment using k-d tree spatial indexing.
Avoids O(N²) brute force — uses scipy KDTree to pre-filter candidate pairs,
then runs precise TCA calculation only on nearby debris.
"""

import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

from scipy.spatial import KDTree
from scipy.optimize import minimize_scalar

from physics.propagator import propagate, propagate_states
from state.fleet import (
    SatelliteObject, DebrisObject, CDMWarning,
    fleet_state
)

logger = logging.getLogger("acm.conjunction")

# ── Thresholds ────────────────────────────────────────────────────────────
CRITICAL_DIST_KM  = 0.100   # 100 m — collision threshold
WARNING_DIST_KM   = 5.0     # 5 km — warning threshold
PREFILTER_DIST_KM = 50.0    # k-d tree initial search radius
LOOKAHEAD_HOURS   = 24      # prediction horizon
LOOKAHEAD_SECS    = LOOKAHEAD_HOURS * 3600
COARSE_STEPS      = 60      # number of coarse time samples for TCA search
FINE_STEPS        = 100     # fine TCA refinement steps


def build_debris_tree(debris_list: List[DebrisObject]) -> Tuple[KDTree, List[str]]:
    """
    Build a k-d tree over current debris positions for fast spatial lookup.
    Returns (tree, ordered_ids).
    """
    if not debris_list:
        return None, []
    positions = np.array([d.r for d in debris_list])
    ids = [d.deb_id for d in debris_list]
    return KDTree(positions), ids


def compute_tca(r_sat: np.ndarray, v_sat: np.ndarray,
                r_deb: np.ndarray, v_deb: np.ndarray,
                lookahead_secs: float = LOOKAHEAD_SECS) -> Tuple[float, float]:
    """
    Find Time of Closest Approach (TCA) and minimum miss distance for a pair.
    Uses coarse sampling followed by scipy minimization refinement.

    Returns:
        (tca_seconds_from_now, min_distance_km)
    """
    # Coarse pass: sample at regular intervals to find rough minimum
    t_coarse = np.linspace(0, lookahead_secs, COARSE_STEPS)
    sat_states = propagate_states(r_sat, v_sat, t_coarse)
    deb_states = propagate_states(r_deb, v_deb, t_coarse)

    distances = [np.linalg.norm(s[0] - d[0])
                 for s, d in zip(sat_states, deb_states)]
    min_idx = int(np.argmin(distances))

    # If the minimum is at the first or last point, TCA is outside window
    if distances[min_idx] > WARNING_DIST_KM * 20:
        return float('inf'), distances[min_idx]

    # Narrow search window around coarse minimum
    t_lo = t_coarse[max(0, min_idx - 1)]
    t_hi = t_coarse[min(len(t_coarse) - 1, min_idx + 1)]

    def neg_distance(t_scalar):
        # We minimize distance (positive value) in [t_lo, t_hi]
        r_s, _ = propagate(r_sat, v_sat, t_scalar)
        r_d, _ = propagate(r_deb, v_deb, t_scalar)
        return float(np.linalg.norm(r_s - r_d))

    result = minimize_scalar(
        neg_distance,
        bounds=(t_lo, t_hi),
        method='bounded',
        options={'xatol': 1.0}  # 1-second accuracy
    )

    tca_t  = result.x
    tca_d  = result.fun
    return tca_t, tca_d


def run_conjunction_assessment(
        satellites: dict,
        debris: dict,
        sim_time: datetime,
        lookahead_secs: float = LOOKAHEAD_SECS) -> List[CDMWarning]:
    """
    Full conjunction assessment for the fleet.
    Uses k-d tree pre-filtering then precise TCA only on candidates.
    Returns list of CDMWarning objects sorted by miss distance.
    """
    warnings: List[CDMWarning] = []

    if not debris:
        return warnings

    debris_list = list(debris.values())
    tree, deb_ids = build_debris_tree(debris_list)

    if tree is None:
        return warnings

    for sat in satellites.values():
        if sat.status == "EOL":
            continue

        # k-d tree query: find all debris within PREFILTER_DIST_KM of sat
        candidate_indices = tree.query_ball_point(sat.r, PREFILTER_DIST_KM * 3)

        # Also query at future position (1 orbit ~90 min ahead) for faster debris
        r_future, v_future = propagate(sat.r, sat.v, 3600.0)
        future_indices = tree.query_ball_point(r_future, PREFILTER_DIST_KM * 5)

        all_indices = list(set(candidate_indices + future_indices))

        if not all_indices:
            continue

        logger.debug(f"{sat.sat_id}: checking {len(all_indices)} debris candidates")

        for idx in all_indices:
            deb = debris_list[idx]

            tca_secs, miss_km = compute_tca(
                sat.r, sat.v, deb.r, deb.v, lookahead_secs
            )

            if tca_secs == float('inf'):
                continue

            if miss_km < WARNING_DIST_KM:
                severity = "CRITICAL" if miss_km < CRITICAL_DIST_KM else "WARNING"
                tca_time = sim_time + timedelta(seconds=tca_secs)

                # Relative velocity at TCA
                r_s, v_s = propagate(sat.r, sat.v, tca_secs)
                r_d, v_d = propagate(deb.r, deb.v, tca_secs)
                rel_vel = float(np.linalg.norm(v_s - v_d))

                warnings.append(CDMWarning(
                    sat_id=sat.sat_id,
                    deb_id=deb.deb_id,
                    tca_time=tca_time,
                    miss_distance_km=round(miss_km, 6),
                    relative_velocity_km_s=round(rel_vel, 4),
                    severity=severity
                ))

                logger.info(
                    f"[{severity}] {sat.sat_id} ↔ {deb.deb_id}: "
                    f"miss={miss_km*1000:.1f}m at {tca_time.isoformat()}"
                )

    # Sort: critical first, then by miss distance
    warnings.sort(key=lambda w: (w.severity != "CRITICAL", w.miss_distance_km))
    return warnings