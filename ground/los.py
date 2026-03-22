"""
ground/los.py
Line-of-sight (LOS) calculator for ground station visibility windows.
Determines if a satellite has radio contact with any ground station,
accounting for Earth's curvature and minimum elevation angle.
"""

import math
import logging
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("acm.ground")

RE = 6378.137  # km


@dataclass
class GroundStation:
    station_id: str
    name: str
    lat_deg: float
    lon_deg: float
    elevation_m: float
    min_elev_angle_deg: float

    @property
    def lat_rad(self) -> float:
        return math.radians(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        return math.radians(self.lon_deg)

    def ecef_position(self) -> np.ndarray:
        """Ground station position in ECEF (approx ECI at t=0) in km."""
        alt_km = self.elevation_m / 1000.0
        r = RE + alt_km
        x = r * math.cos(self.lat_rad) * math.cos(self.lon_rad)
        y = r * math.cos(self.lat_rad) * math.sin(self.lon_rad)
        z = r * math.sin(self.lat_rad)
        return np.array([x, y, z])


# ── Ground Station Network (from problem spec) ────────────────────────────────
GROUND_STATIONS: List[GroundStation] = [
    GroundStation("GS-001", "ISTRAC_Bengaluru",      13.0333,  77.5167,  820, 5.0),
    GroundStation("GS-002", "Svalbard_Sat_Station",  78.2297,  15.4077,  400, 5.0),
    GroundStation("GS-003", "Goldstone_Tracking",    35.4266,-116.8900, 1000,10.0),
    GroundStation("GS-004", "Punta_Arenas",         -53.1500, -70.9167,   30, 5.0),
    GroundStation("GS-005", "IIT_Delhi_Ground_Node", 28.5450,  77.1926,  225,15.0),
    GroundStation("GS-006", "McMurdo_Station",      -77.8463, 166.6682,   10, 5.0),
]


def elevation_angle(r_sat: np.ndarray, gs: GroundStation) -> float:
    """
    Compute elevation angle (degrees) of satellite as seen from ground station.
    Uses geometric line-of-sight; ignores Earth rotation for short timescales.
    Returns elevation angle in degrees. Negative = below horizon.
    """
    r_gs = gs.ecef_position()

    # Vector from GS to satellite
    r_rel = r_sat - r_gs
    dist = np.linalg.norm(r_rel)

    if dist < 1e-6:
        return 90.0

    # Nadir unit vector at GS (pointing toward Earth center)
    r_gs_mag = np.linalg.norm(r_gs)
    zenith = r_gs / r_gs_mag  # unit vector pointing up from GS

    # Elevation = 90 - angle between zenith and (sat - GS)
    cos_angle = np.dot(zenith, r_rel / dist)
    elev_rad = math.asin(max(-1.0, min(1.0, cos_angle)))
    return math.degrees(elev_rad)


def has_line_of_sight(r_sat: np.ndarray,
                      stations: List[GroundStation] = GROUND_STATIONS) -> Tuple[bool, Optional[str]]:
    """
    Check if the satellite has LOS to at least one ground station.
    Returns (has_los: bool, station_name: Optional[str])
    """
    for gs in stations:
        elev = elevation_angle(r_sat, gs)
        if elev >= gs.min_elev_angle_deg:
            return True, gs.station_id
    return False, None


def find_next_los_window(
        r_sat: np.ndarray, v_sat: np.ndarray,
        sim_time,
        max_search_seconds: float = 7200.0,
        dt: float = 30.0
) -> Optional[float]:
    """
    Scan forward in time to find the next window when the satellite
    has LOS to any ground station.
    Returns seconds from now when LOS begins (or None if not found).
    """
    from physics.propagator import rk4_step

    state = np.concatenate([r_sat, v_sat])
    t = 0.0

    while t < max_search_seconds:
        r_current = state[:3]
        los, _ = has_line_of_sight(r_current)
        if los:
            return t
        state = rk4_step(state, dt)
        t += dt

    return None  # No LOS window found in search window


def find_last_los_before_time(
        r_sat: np.ndarray, v_sat: np.ndarray,
        target_seconds: float,
        dt: float = 30.0
) -> Optional[float]:
    """
    Find the last time (seconds from now) before target_seconds
    when the satellite was in LOS with any ground station.
    Used to schedule burns before blackout zones.
    """
    from physics.propagator import rk4_step

    state = np.concatenate([r_sat, v_sat])
    last_los_time = None
    t = 0.0

    while t < target_seconds:
        r_current = state[:3]
        los, station = has_line_of_sight(r_current)
        if los:
            last_los_time = t
        state = rk4_step(state, dt)
        t += dt

    return last_los_time


def eci_to_latlon(r_eci: np.ndarray, gmst_rad: float = 0.0) -> Tuple[float, float, float]:
    """
    Convert ECI position to geodetic latitude, longitude, altitude.
    gmst_rad: Greenwich Mean Sidereal Time in radians (0 for approximate)
    Returns (lat_deg, lon_deg, alt_km)
    """
    x, y, z = r_eci
    r_mag = np.linalg.norm(r_eci)

    # Longitude (approximate, ignoring Earth rotation for snapshot)
    lon_rad = math.atan2(y, x) - gmst_rad
    lon_deg = math.degrees(lon_rad) % 360
    if lon_deg > 180:
        lon_deg -= 360

    # Latitude (geocentric, sufficient for dashboard)
    lat_rad = math.asin(z / r_mag)
    lat_deg = math.degrees(lat_rad)

    alt_km = r_mag - RE
    return lat_deg, lon_deg, alt_km
