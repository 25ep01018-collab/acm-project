"""
api/visualization.py
GET /api/visualization/snapshot
Returns real satellite positions (converted from ECI state vectors to lat/lon)
plus debris cloud and CDM warnings for the Orbital Insight dashboard.
"""

import math
import datetime as dt
from fastapi import APIRouter
from state.fleet import fleet_state

router = APIRouter()

# Earth rotation rate (rad/s) — needed to convert ECI → geographic lat/lon
EARTH_ROT_RAD_S = 7.2921150e-5
RE = 6378.137  # km

# Fixed reference epoch for GMST approximation
SIM_EPOCH = dt.datetime(2026, 3, 12, 8, 0, 0, tzinfo=dt.timezone.utc)


def eci_to_latlon(r, sim_time):
    """
    Convert ECI position vector [x, y, z] (km) to (lat_deg, lon_deg, alt_km).
    Accounts for Earth's rotation since simulation epoch.
    """
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    r_mag = math.sqrt(x*x + y*y + z*z)

    # Latitude (geocentric)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z / r_mag))))

    # Longitude — subtract Earth rotation since epoch
    elapsed = (sim_time - SIM_EPOCH).total_seconds()
    gmst = (elapsed * EARTH_ROT_RAD_S) % (2 * math.pi)
    lon = math.degrees(math.atan2(y, x)) - math.degrees(gmst)

    # Normalize to [-180, 180]
    lon = ((lon + 180) % 360) - 180

    alt = r_mag - RE
    return round(lat, 4), round(lon, 4), round(alt, 2)


@router.get("/visualization/snapshot")
def get_snapshot():
    sim_time = fleet_state.sim_time

    # ── Satellites ────────────────────────────────────────────────────────
    satellites = []
    for sat in fleet_state.satellites.values():
        lat, lon, alt = eci_to_latlon(sat.r, sim_time)

        cdm_count = sum(
            1 for w in fleet_state.cdm_warnings
            if w.sat_id == sat.sat_id
        )

        satellites.append({
            "id":          sat.sat_id,
            "lat":         lat,
            "lon":         lon,
            "alt_km":      alt,
            "fuel_kg":     round(sat.fuel_kg, 2),
            "fuel_pct":    round(sat.fuel_fraction * 100, 1),
            "status":      sat.status,
            "in_slot":     sat.in_station_box(),
            "cdm_count":   cdm_count,
            "dv_total_ms": round(sat.total_dv_used * 1000, 2),
        })

    # ── Debris cloud — compact [id, lat, lon, alt] tuples ────────────────
    debris_cloud = []
    for deb in fleet_state.debris.values():
        lat, lon, alt = eci_to_latlon(deb.r, sim_time)
        debris_cloud.append([deb.deb_id, lat, lon, alt])

    # ── CDM warnings ──────────────────────────────────────────────────────
    cdm_warnings = [
        {
            "sat_id":       w.sat_id,
            "deb_id":       w.deb_id,
            "tca":          w.tca_time.isoformat(),
            "miss_km":      w.miss_distance_km,
            "rel_vel_kmps": w.relative_velocity_km_s,
            "severity":     w.severity,
        }
        for w in fleet_state.cdm_warnings[:50]
    ]

    # ── Fleet-wide stats ──────────────────────────────────────────────────
    total_fuel    = sum(s.fuel_kg       for s in fleet_state.satellites.values())
    total_dv      = sum(s.total_dv_used for s in fleet_state.satellites.values())
    nominal_count = sum(1 for s in fleet_state.satellites.values() if s.in_station_box())
    eol_count     = sum(1 for s in fleet_state.satellites.values() if s.status == "EOL")

    return {
        "timestamp":    sim_time.isoformat().replace("+00:00", "Z"),
        "satellites":   satellites,
        "debris_cloud": debris_cloud,
        "cdm_warnings": cdm_warnings,
        "fleet_stats": {
            "total_satellites":        len(fleet_state.satellites),
            "total_debris_tracked":    len(fleet_state.debris),
            "nominal_count":           nominal_count,
            "eol_count":               eol_count,
            "active_cdm_warnings":     fleet_state.active_cdm_count(),
            "fleet_fuel_remaining_kg": round(total_fuel, 1),
            "fleet_dv_used_ms":        round(total_dv * 1000, 2),
            "sim_time":                sim_time.isoformat(),
        }
    }