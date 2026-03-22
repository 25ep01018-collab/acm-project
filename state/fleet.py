"""
state/fleet.py
Central in-memory state store for the entire constellation and debris field.
Tracks satellite objects, debris objects, maneuver queues, and simulation time.
"""

import math
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("acm.state")

# ── Spacecraft physical constants ──────────────────────────────────────────────
DRY_MASS_KG        = 500.0   # kg
INITIAL_FUEL_KG    = 50.0    # kg
INITIAL_WET_MASS   = DRY_MASS_KG + INITIAL_FUEL_KG  # 550 kg
ISP_S              = 300.0   # specific impulse (seconds)
G0_M_S2            = 9.80665 # standard gravity (m/s²)
G0_KM_S2           = G0_M_S2 / 1000.0
MAX_DV_PER_BURN    = 0.015   # km/s (15 m/s)
THRUSTER_COOLDOWN  = 600.0   # seconds
FUEL_EOL_FRACTION  = 0.05    # 5% remaining → graveyard orbit
STATION_BOX_KM     = 10.0    # nominal slot radius (km)

# ── Earth constants ─────────────────────────────────────────────────────────
MU      = 398600.4418  # km³/s²
RE      = 6378.137     # km
J2      = 1.08263e-3


@dataclass
class BurnCommand:
    burn_id: str
    burn_time: datetime
    delta_v: np.ndarray          # [dvx, dvy, dvz] in km/s (ECI)
    executed: bool = False


@dataclass
class SatelliteObject:
    sat_id: str
    r: np.ndarray                # position ECI (km)
    v: np.ndarray                # velocity ECI (km/s)
    fuel_kg: float = INITIAL_FUEL_KG
    dry_mass: float = DRY_MASS_KG
    status: str = "NOMINAL"      # NOMINAL | EVADING | RECOVERING | EOL
    nominal_slot_r: Optional[np.ndarray] = None  # ideal slot position
    nominal_slot_v: Optional[np.ndarray] = None
    last_burn_time: Optional[datetime] = None
    maneuver_queue: List[BurnCommand] = field(default_factory=list)
    service_outage_seconds: float = 0.0
    total_dv_used: float = 0.0   # km/s consumed total
    collisions_avoided: int = 0

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.fuel_kg

    @property
    def fuel_fraction(self) -> float:
        return self.fuel_kg / INITIAL_FUEL_KG

    @property
    def is_eol(self) -> bool:
        return self.fuel_fraction <= FUEL_EOL_FRACTION

    def distance_from_slot(self) -> float:
        """Distance in km from nominal orbital slot."""
        if self.nominal_slot_r is None:
            return 0.0
        return float(np.linalg.norm(self.r - self.nominal_slot_r))

    def in_station_box(self) -> bool:
        return self.distance_from_slot() <= STATION_BOX_KM


@dataclass
class DebrisObject:
    deb_id: str
    r: np.ndarray   # position ECI (km)
    v: np.ndarray   # velocity ECI (km/s)


@dataclass
class CDMWarning:
    sat_id: str
    deb_id: str
    tca_time: datetime
    miss_distance_km: float
    relative_velocity_km_s: float
    severity: str    # WARNING | CRITICAL


class FleetState:
    """
    Central singleton state store.
    All API handlers and physics modules share this object.
    """

    def __init__(self):
        self.satellites:  Dict[str, SatelliteObject] = {}
        self.debris:      Dict[str, DebrisObject]    = {}
        self.cdm_warnings: List[CDMWarning]          = []
        self.sim_time: datetime = datetime.now(timezone.utc)
        self.maneuver_log: List[dict] = []
        self._lock_available = False

    # ── Initialisation ─────────────────────────────────────────────────────

    def initialize(self):
        """Seed the fleet with 50 satellites in LEO orbits."""
        logger.info("Seeding initial constellation (50 satellites)...")
        self.sim_time = datetime(2026, 3, 12, 8, 0, 0, tzinfo=timezone.utc)

        for i in range(50):
            # Distribute satellites in slightly different orbital planes
            # Semi-major axis ~6778 km (400 km altitude)
            altitude_km = 400.0 + (i % 10) * 5.0
            a = RE + altitude_km
            inc_deg = 53.0 + (i // 10) * 5.0
            raan_deg = (i * 7.2) % 360.0
            ma_deg   = (i * 7.2) % 360.0

            r, v = keplerian_to_eci(
                a=a,
                e=0.0001,
                i=math.radians(inc_deg),
                raan=math.radians(raan_deg),
                argp=math.radians(0.0),
                M=math.radians(ma_deg),
            )
            sat_id = f"SAT-Alpha-{i+1:02d}"
            sat = SatelliteObject(
                sat_id=sat_id,
                r=r.copy(),
                v=v.copy(),
                nominal_slot_r=r.copy(),
                nominal_slot_v=v.copy(),
            )
            self.satellites[sat_id] = sat

        logger.info(f"Initialized {len(self.satellites)} satellites.")

    # ── Helpers ────────────────────────────────────────────────────────────

    def upsert_object(self, obj_id: str, obj_type: str,
                      r: np.ndarray, v: np.ndarray):
        """Insert or update a satellite or debris object from telemetry."""
        if obj_type == "DEBRIS":
            self.debris[obj_id] = DebrisObject(deb_id=obj_id, r=r, v=v)
        elif obj_type == "SATELLITE":
            if obj_id in self.satellites:
                self.satellites[obj_id].r = r
                self.satellites[obj_id].v = v
            else:
                self.satellites[obj_id] = SatelliteObject(
                    sat_id=obj_id, r=r, v=v,
                    nominal_slot_r=r.copy(), nominal_slot_v=v.copy()
                )

    def active_cdm_count(self) -> int:
        return len([w for w in self.cdm_warnings
                    if w.severity in ("WARNING", "CRITICAL")])

    def log_maneuver(self, sat_id: str, burn_id: str,
                     burn_time: datetime, dv: np.ndarray,
                     fuel_remaining: float):
        self.maneuver_log.append({
            "sat_id": sat_id,
            "burn_id": burn_id,
            "burn_time": burn_time.isoformat(),
            "delta_v_kmps": dv.tolist(),
            "dv_magnitude_ms": float(np.linalg.norm(dv) * 1000),
            "fuel_remaining_kg": round(fuel_remaining, 4),
        })


# ── Orbital mechanics helpers ───────────────────────────────────────────────

def keplerian_to_eci(a, e, i, raan, argp, M):
    """
    Convert Keplerian elements to ECI state vector.
    a    - semi-major axis (km)
    e    - eccentricity
    i    - inclination (rad)
    raan - RAAN (rad)
    argp - argument of perigee (rad)
    M    - mean anomaly (rad)
    Returns r (km), v (km/s) as numpy arrays.
    """
    # Solve Kepler's equation for eccentric anomaly E
    E = M
    for _ in range(100):
        dE = (M - (E - e * math.sin(E))) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break

    # True anomaly
    nu = 2 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2)
    )

    # Distance
    r_mag = a * (1 - e * math.cos(E))

    # Position and velocity in perifocal frame
    r_pf = np.array([r_mag * math.cos(nu), r_mag * math.sin(nu), 0.0])
    p = a * (1 - e**2)
    v_pf = np.array([
        -math.sqrt(MU / p) * math.sin(nu),
         math.sqrt(MU / p) * (e + math.cos(nu)),
         0.0
    ])

    # Rotation matrix: perifocal → ECI
    cos_O, sin_O = math.cos(raan), math.sin(raan)
    cos_w, sin_w = math.cos(argp), math.sin(argp)
    cos_i, sin_i = math.cos(i),    math.sin(i)

    R = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i,
         -cos_O*sin_w - sin_O*cos_w*cos_i,
          sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i,
         -sin_O*sin_w + cos_O*cos_w*cos_i,
         -cos_O*sin_i],
        [sin_w*sin_i,  cos_w*sin_i,  cos_i]
    ])

    return R @ r_pf, R @ v_pf


# ── Global singleton ────────────────────────────────────────────────────────
fleet_state = FleetState()