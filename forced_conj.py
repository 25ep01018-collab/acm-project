import json
import math
import urllib.request
import time

BASE = "http://localhost:8000"

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return json.loads(r.read())

# ── Step 1: Get real satellite ECI positions from fleet state
# We seed debris at EXACT satellite positions + tiny offset
# This guarantees miss distance < 100m immediately

print("Step 1: Reading satellite positions...")
snap = get("/api/visualization/snapshot")
sats = snap["satellites"]
print(f"  Found {len(sats)} satellites")

# ── Step 2: Convert lat/lon/alt → ECI properly
RE = 6378.137
MU = 398600.4418

def latlon_to_eci(lat_deg, lon_deg, alt_km):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r   = RE + alt_km
    x   = r * math.cos(lat) * math.cos(lon)
    y   = r * math.cos(lat) * math.sin(lon)
    z   = r * math.sin(lat)
    return x, y, z

def orbital_velocity(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    v = math.sqrt(MU / r)
    # Velocity perpendicular to position in equatorial-ish direction
    vx = -y / math.sqrt(x*x + y*y) * v
    vy =  x / math.sqrt(x*x + y*y) * v
    vz = 0.0
    return vx, vy, vz

# ── Step 3: Place debris at EXACTLY satellite positions + 0.05km offset
print("Step 2: Creating debris at exact satellite positions...")
debris_list = []
deb_id = 0

for sat in sats:
    sx, sy, sz = latlon_to_eci(sat["lat"], sat["lon"], sat.get("alt_km", 400))
    vx, vy, vz = orbital_velocity(sx, sy, sz)

    # Place 3 debris pieces per satellite
    # Offset 1: 0.05 km ahead in velocity direction (50m — within collision threshold)
    debris_list.append({
        "id": f"THREAT-{deb_id:03d}",
        "type": "DEBRIS",
        "r": {"x": round(sx + 0.050, 4),
              "y": round(sy + 0.050, 4),
              "z": round(sz,         4)},
        "v": {"x": round(vx + 0.001, 6),
              "y": round(vy - 0.001, 6),
              "z": round(vz + 0.001, 6)}
    })
    deb_id += 1

    # Offset 2: 0.03 km (30m — definite collision)
    debris_list.append({
        "id": f"THREAT-{deb_id:03d}",
        "type": "DEBRIS",
        "r": {"x": round(sx + 0.030, 4),
              "y": round(sy - 0.030, 4),
              "z": round(sz + 0.010, 4)},
        "v": {"x": round(vx - 0.002, 6),
              "y": round(vy + 0.002, 6),
              "z": round(vz,         6)}
    })
    deb_id += 1

    # Offset 3: 1.0 km (warning range)
    debris_list.append({
        "id": f"WARN-{deb_id:03d}",
        "type": "DEBRIS",
        "r": {"x": round(sx + 1.0, 4),
              "y": round(sy,        4),
              "z": round(sz,        4)},
        "v": {"x": round(vx, 6),
              "y": round(vy, 6),
              "z": round(vz, 6)}
    })
    deb_id += 1

print(f"  Created {len(debris_list)} debris objects")

# ── Step 4: Send to telemetry
print("Step 3: Sending to /api/telemetry...")
resp = post("/api/telemetry", {
    "timestamp": "2026-03-12T08:00:00.000Z",
    "objects": debris_list
})
print(f"  Processed : {resp['processed_count']}")
print(f"  CDM warnings immediately: {resp['active_cdm_warnings']}")

# ── Step 5: Run simulation steps and watch
print("\nStep 4: Running simulation (watch your browser!)...")
print("-" * 55)

total_burns = 0
for i in range(40):
    r = post("/api/simulate/step", {"step_seconds": 30})
    c = r.get("collisions_detected", 0)
    m = r.get("maneuvers_executed", 0)
    t = r.get("new_timestamp", "")[11:19]
    total_burns += m

    line = f"  [{t}]"
    if c > 0: line += f"  COLLISION x{c} !!!"
    if m > 0: line += f"  BURN FIRED x{m}"
    if c == 0 and m == 0: line += "  nominal"
    print(line)
    time.sleep(0.15)

print("-" * 55)
print(f"\nTotal burns fired: {total_burns}")

# ── Step 6: Final status
print("\nFinal fleet status:")
snap  = get("/api/visualization/snapshot")
fs    = snap["fleet_stats"]
sats  = snap["satellites"]
warns = snap.get("cdm_warnings", [])

evading    = [s for s in sats if s["status"] == "EVADING"]
recovering = [s for s in sats if s["status"] == "RECOVERING"]
eol        = [s for s in sats if s["status"] == "EOL"]

print(f"  CDM active    : {fs.get('active_cdm_warnings')}")
print(f"  Fuel left     : {fs.get('fleet_fuel_remaining_kg')} kg")
print(f"  Total dV used : {fs.get('fleet_dv_used_ms')} m/s")
print(f"  EVADING       : {len(evading)} sats  {[s['id'].replace('SAT-Alpha-','#') for s in evading]}")
print(f"  RECOVERING    : {len(recovering)} sats")
print(f"  EOL           : {len(eol)} sats")

if warns:
    print(f"\n  Active warnings ({len(warns)}):")
    for w in warns[:8]:
        print(f"    {w['sat_id']} <-> {w['deb_id']} | "
              f"{w['miss_km']*1000:.1f}m | {w['severity']}")

if total_burns == 0:
    print("\n  No burns fired — the conjunction engine may need")
    print("  a simulation step to process. Try running again.")
else:
    print(f"\n  SUCCESS! {total_burns} evasion burns executed.")
    print("  Refresh http://localhost:8000 to see satellites in colour!")