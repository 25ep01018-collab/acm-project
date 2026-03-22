import json
import math
import random
import time
import urllib.request

BASE = "http://localhost:8000"

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())

print("Fetching satellite positions...")
snap = get("/api/visualization/snapshot")
sats = snap.get("satellites", [])
print(f"Found {len(sats)} satellites")

RE = 6378.137
debris_list = []
deb_id = 0

for i, sat in enumerate(sats[:25]):
    lat = math.radians(sat["lat"])
    lon = math.radians(sat["lon"])
    alt = sat.get("alt_km", 400)
    r   = RE + alt

    sx = r * math.cos(lat) * math.cos(lon)
    sy = r * math.cos(lat) * math.sin(lon)
    sz = r * math.sin(lat)

    v_mag = math.sqrt(398600.4418 / r)

    for j in range(4):
        offset = random.uniform(0.05, 1.5)
        angle  = random.uniform(0, 2 * math.pi)
        dv     = random.uniform(-0.05, 0.05)

        debris_list.append({
            "id":   f"CLOSE-{deb_id:04d}",
            "type": "DEBRIS",
            "r": {
                "x": round(sx + offset * math.cos(angle), 4),
                "y": round(sy + offset * math.sin(angle), 4),
                "z": round(sz + random.uniform(-0.3, 0.3), 4)
            },
            "v": {
                "x": round(-sy / r * v_mag + dv, 6),
                "y": round( sx / r * v_mag + dv, 6),
                "z": round(dv * 0.1, 6)
            }
        })
        deb_id += 1

print(f"Seeding {len(debris_list)} targeted debris objects...")
resp = post("/api/telemetry", {
    "timestamp": "2026-03-12T08:00:00.000Z",
    "objects": debris_list
})
print(f"Seeded: {resp['processed_count']}")
print(f"CDM warnings: {resp['active_cdm_warnings']}")

print("\nRunning 30 simulation steps...")
for i in range(30):
    r = post("/api/simulate/step", {"step_seconds": 30})
    c = r.get("collisions_detected", 0)
    m = r.get("maneuvers_executed", 0)
    t = r.get("new_timestamp", "")[11:19]
    line = f"  Step {i+1:02d} | {t}"
    if c > 0: line += f" | COLLISION x{c}"
    if m > 0: line += f" | {m} BURN(S) FIRED"
    print(line)
    time.sleep(0.2)

print("\nFinal status:")
snap = get("/api/visualization/snapshot")
fs   = snap.get("fleet_stats", {})
sats = snap.get("satellites", [])
warns = snap.get("cdm_warnings", [])
print(f"  CDM warnings  : {fs.get('active_cdm_warnings')}")
print(f"  Fuel remaining: {fs.get('fleet_fuel_remaining_kg')} kg")
print(f"  Total dV used : {fs.get('fleet_dv_used_ms')} m/s")
evading    = [s for s in sats if s["status"] == "EVADING"]
recovering = [s for s in sats if s["status"] == "RECOVERING"]
print(f"  Evading       : {len(evading)}")
print(f"  Recovering    : {len(recovering)}")
if warns:
    print("\n  Top warnings:")
    for w in warns[:5]:
        print(f"    {w['sat_id']} <-> {w['deb_id']} | {w['miss_km']*1000:.0f}m | {w['severity']}")
print("\nRefresh http://localhost:8000 to see satellites evading!")