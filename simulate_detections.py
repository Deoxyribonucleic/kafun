"""
Simulate crowdsourced pollen detections around Ritsumeikan OIC campus.
Points cluster around fake "tree stands" (the way real photos pile up near trees),
each record matching what predict_gate.py will eventually output.
Output: detections.json
"""
import json, random, math, datetime

random.seed(42)  # reproducible; remove for fresh random each run

# OIC campus approximate center
CENTER_LAT, CENTER_LON = 34.8085, 135.5605

# fake tree stands: (name, species, lat, lon, n_detections, spread_in_meters)
STANDS = [
    ("cedar row north",   "sugi",   34.8098, 135.5598, 60, 35),
    ("cedar stand east",  "sugi",   34.8079, 135.5631, 45, 30),
    ("cypress west",      "hinoki", 34.8072, 135.5589, 50, 30),
    ("cypress courtyard", "hinoki", 34.8090, 135.5616, 40, 25),
    ("mixed broadleaf",   "other",  34.8083, 135.5605, 55, 60),  # spread out
]

# ~1 degree latitude ≈ 111,320 m; longitude scaled by cos(lat)
def meters_to_deg(m_north, m_east, lat):
    dlat = m_north / 111320.0
    dlon = m_east / (111320.0 * math.cos(math.radians(lat)))
    return dlat, dlon

def conf_for(species):
    # realistic confidences: conifers a bit lower (harder), other higher
    base = {"sugi": 0.86, "hinoki": 0.84, "other": 0.95}[species]
    return round(min(0.99, max(0.70, random.gauss(base, 0.06))), 2)

detections = []
now = datetime.datetime.now()
for name, species, lat, lon, n, spread in STANDS:
    for _ in range(n):
        # gaussian scatter around the stand center
        m_north = random.gauss(0, spread)
        m_east  = random.gauss(0, spread)
        dlat, dlon = meters_to_deg(m_north, m_east, lat)
        # random timestamp within the last 30 days
        ts = now - datetime.timedelta(days=random.uniform(0, 30),
                                      hours=random.uniform(0, 24))
        detections.append({
            "lat": round(lat + dlat, 6),
            "lon": round(lon + dlon, 6),
            "species": species,
            "confidence": conf_for(species),
            "timestamp": ts.isoformat(timespec="seconds"),
        })

random.shuffle(detections)
with open("detections.json", "w") as f:
    json.dump(detections, f, indent=2)

# summary
from collections import Counter
counts = Counter(d["species"] for d in detections)
print(f"wrote {len(detections)} detections -> detections.json")
for sp in ("sugi", "hinoki", "other"):
    print(f"  {sp:8s}: {counts.get(sp,0)}")
print(f"center: {CENTER_LAT}, {CENTER_LON}  | stands: {len(STANDS)}")
