"""
test_api.py
Run with: python test_api.py
"""

import json
import math
import random
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

BASE = "http://localhost:8000"

def post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
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

def future_time(hours=6):
    """Get a timestamp N hours from sim start so burns are in the future."""
    t = datetime(2026, 3, 12, 8, 0, 0, tzinfo=timezone.utc) + timedelta(hours=hours)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")

def test_telemetry():
    print("\n=== TEST 1: POST /api/telemetry ===")
    resp = post("/api/telemetry", {
        "timestamp": "2026-03-12T08:00:00.000Z",
        "objects": [
            {
                "id": "DEB-99421",
                "type": "DEBRIS",
                "r": {"x": 4500.2, "y": -2100.5, "z": 4800.1},
                "v": {"x": -1.25,  "y": 6.84,    "z": 3.12}
            }
        ]
    })
    print(json.dumps(resp, indent=2))
    assert resp["status"] == "ACK"
    print("PASS")

def test_maneuver():
    print("\n=== TEST 2: POST /api/maneuver/schedule ===")
    resp = post("/api/maneuver/schedule", {
        "satelliteId": "SAT-Alpha-01",
        "maneuver_sequence": [
            {
                "burn_id": "EVASION_BURN_1",
                "burnTime": future_time(hours=6),
                "deltaV_vector": {"x": 0.002, "y": 0.010, "z": -0.001}
            },
            {
                "burn_id": "RECOVERY_BURN_1",
                "burnTime": future_time(hours=8),
                "deltaV_vector": {"x": -0.002, "y": -0.010, "z": 0.001}
            }
        ]
    })
    print(json.dumps(resp, indent=2))
    assert resp["status"] == "SCHEDULED"
    print("PASS")

def test_simulate():
    print("\n=== TEST 3: POST /api/simulate/step ===")
    resp = post("/api/simulate/step", {"step_seconds": 60})
    print(json.dumps(resp, indent=2))
    assert resp["status"] == "STEP_COMPLETE"
    print("PASS")

def test_snapshot():
    print("\n=== TEST 4: GET /api/visualization/snapshot ===")
    resp = get("/api/visualization/snapshot")
    print(f"  satellites : {len(resp.get('satellites', []))}")
    print(f"  debris     : {len(resp.get('debris_cloud', []))}")
    print(f"  cdm_warnings: {len(resp.get('cdm_warnings', []))}")
    for k, v in resp.get("fleet_stats", {}).items():
        print(f"  {k}: {v}")
    assert "satellites" in resp
    print("PASS")

def seed_debris():
    print("\n=== SEEDING 500 DEBRIS OBJECTS ===")
    debris_list = []
    for i in range(500):
        angle = random.uniform(0, 2 * math.pi)
        inc   = math.radians(random.uniform(40, 70))
        r_mag = 6778 + random.uniform(-100, 100)
        if i < 20:
            r_mag = 6778 + random.uniform(-5, 5)
        debris_list.append({
            "id":   f"DEB-{i:05d}",
            "type": "DEBRIS",
            "r": {
                "x": round(r_mag * math.cos(angle) * math.cos(inc), 3),
                "y": round(r_mag * math.sin(angle), 3),
                "z": round(r_mag * math.cos(angle) * math.sin(inc), 3)
            },
            "v": {
                "x": round(random.uniform(-1.0, 1.0), 4),
                "y": round(7.8 + random.uniform(-0.5, 0.5), 4),
                "z": round(random.uniform(-0.2, 0.2), 4)
            }
        })
    resp = post("/api/telemetry", {
        "timestamp": "2026-03-12T08:00:00.000Z",
        "objects": debris_list
    })
    print(f"  Debris seeded    : {resp['processed_count']}")
    print(f"  CDM warnings now : {resp['active_cdm_warnings']}")
    print("  Refresh browser — debris dots should appear on map!")

def run_simulation(steps=20, step_seconds=60):
    print(f"\n=== RUNNING SIMULATION ({steps} steps x {step_seconds}s) ===")
    for i in range(steps):
        result = post("/api/simulate/step", {"step_seconds": step_seconds})
        t          = result.get("new_timestamp", "")
        collisions = result.get("collisions_detected", 0)
        maneuvers  = result.get("maneuvers_executed", 0)
        col_warn   = " <-- COLLISION!" if collisions > 0 else ""
        man_note   = f" | {maneuvers} burn(s) fired" if maneuvers > 0 else ""
        print(f"  Step {i+1:02d} | {t[11:19]} UTC | "
              f"collisions={collisions}{col_warn}{man_note}")
        time.sleep(0.3)
    print("\nDone! Check browser for updated satellite positions.")

if __name__ == "__main__":
    print("=" * 50)
    print("  ACM API Test Suite")
    print(f"  Target: {BASE}")
    print("=" * 50)
    try:
        test_telemetry()
        time.sleep(0.3)
        test_maneuver()
        time.sleep(0.3)
        test_simulate()
        time.sleep(0.3)
        test_snapshot()

        print("\n" + "=" * 50)
        print("  All 4 basic tests PASSED!")
        print("=" * 50)

        print("\nOptions:")
        print("  1. Seed 500 debris objects")
        print("  2. Run simulation 20 steps")
        print("  3. Both (recommended)")
        print("  4. Skip")
        choice = input("\nEnter choice (1/2/3/4): ").strip()

        if choice in ("1", "3"):
            seed_debris()
            time.sleep(1)
        if choice in ("2", "3"):
            run_simulation(steps=20, step_seconds=60)

        print("\nOpen http://localhost:8000 to see the dashboard.")

    except urllib.error.URLError as e:
        print(f"\nCONNECTION ERROR: {e}")
        print("Make sure uvicorn is running: uvicorn main:app --reload")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except KeyboardInterrupt:
        print("\nStopped.")