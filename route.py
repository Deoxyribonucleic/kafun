import osmnx as ox, networkx as nx, folium, json, math, numpy as np
from folium.plugins import HeatMap
from shapely import wkt

CAMPUS  = (34.8085, 135.5605)   # start
STATION = (34.8157, 135.5630)   # end (JR Ibaraki) -- adjust if marker was off
LAMBDA  = 1.0     # avoidance strength: 0 = shortest; ~1 = max-pollen edge feels ~2x its length
BW_M    = 90.0    # pollen influence radius (metres)

G = ox.load_graphml("oic_walk.graphml")

# some versions store edge geometry as WKT strings after reload; re-parse to real geometry
for _, _, data in G.edges(data=True):
    geom = data.get("geometry")
    if isinstance(geom, str):
        try: data["geometry"] = wkt.loads(geom)
        except Exception: pass

# --- allergenic detections ---
with open("detections.json") as f:
    dets = [d for d in json.load(f) if d["species"] in ("sugi", "hinoki")]
if not dets:
    print("WARNING: no sugi/hinoki detections in detections.json — pollen cost will be 0 everywhere.")
plat = np.array([d["lat"] for d in dets]) if dets else np.array([])
plon = np.array([d["lon"] for d in dets]) if dets else np.array([])
w    = np.array([d["confidence"] for d in dets]) if dets else np.array([])
MLAT = 111320.0
def MLON(lat): return 111320.0 * math.cos(math.radians(lat))

def pollen_at(lat, lon):
    if len(plat) == 0:
        return 0.0
    dy = (plat - lat) * MLAT
    dx = (plon - lon) * MLON(lat)
    return float(np.sum(w * np.exp(-0.5 * (dx*dx + dy*dy) / (BW_M*BW_M))))

# --- pass 1: pollen at each edge midpoint, track max for normalization ---
pmax = 0.0
for u, v, k, data in G.edges(keys=True, data=True):
    y = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
    x = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
    p = pollen_at(y, x)
    data["pollen"] = p
    if p > pmax: pmax = p
pmax = pmax or 1.0

# --- pass 2: costs. plain = distance; combo = distance scaled up by normalized pollen ---
for u, v, k, data in G.edges(keys=True, data=True):
    length = data.get("length", 1.0)
    pnorm  = data["pollen"] / pmax           # 0..1
    data["plain"] = length
    data["combo"] = length * (1.0 + LAMBDA * pnorm)

# --- endpoints ---
orig = ox.nearest_nodes(G, CAMPUS[1], CAMPUS[0])
dest = ox.nearest_nodes(G, STATION[1], STATION[0])
print(f"start node @ {G.nodes[orig]['y']:.5f}, {G.nodes[orig]['x']:.5f}  (wanted {CAMPUS[0]}, {CAMPUS[1]})")
print(f"end   node @ {G.nodes[dest]['y']:.5f}, {G.nodes[dest]['x']:.5f}  (wanted {STATION[0]}, {STATION[1]})")

# --- A* with admissible straight-line distance heuristic ---
def heuristic(n1, n2):
    y1, x1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
    y2, x2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
    dy = (y1 - y2) * MLAT
    dx = (x1 - x2) * MLON((y1 + y2) / 2)
    return math.sqrt(dx*dx + dy*dy)

route_plain = nx.astar_path(G, orig, dest, heuristic=heuristic, weight="plain")
route_avoid = nx.astar_path(G, orig, dest, heuristic=heuristic, weight="combo")

def stats(route):
    dist = sum(min(d["length"] for d in G[u][v].values()) for u, v in zip(route[:-1], route[1:]))
    poll = sum(min(d["pollen"] for d in G[u][v].values()) for u, v in zip(route[:-1], route[1:]))
    return dist, poll

d0, p0 = stats(route_plain)
d1, p1 = stats(route_avoid)
print(f"\nshortest : {d0:6.0f} m   pollen {p0:8.1f}")
print(f"avoiding : {d1:6.0f} m   pollen {p1:8.1f}   (+{d1-d0:.0f} m, "
      f"{100*(1-p1/max(p0,1e-9)):+.0f}% pollen)")

# --- draw the FULL edge geometry (curves), not just node-to-node straight lines ---
def route_latlon(route):
    pts = []
    for u, v in zip(route[:-1], route[1:]):
        data = min(G[u][v].values(), key=lambda d: d.get("length", 1))
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "xy"):
            xs, ys = geom.xy
            seg = list(zip(ys, xs))
            # keep orientation consistent with travel direction
            if pts and seg and math.dist(pts[-1], seg[0]) > math.dist(pts[-1], seg[-1]):
                seg = seg[::-1]
            pts += seg
        else:
            pts.append((G.nodes[u]["y"], G.nodes[u]["x"]))
            pts.append((G.nodes[v]["y"], G.nodes[v]["x"]))
    return pts

mid = ((CAMPUS[0]+STATION[0])/2, (CAMPUS[1]+STATION[1])/2)
m = folium.Map(location=mid, zoom_start=16, tiles="CartoDB positron")
if dets:
    HeatMap([[d["lat"], d["lon"], d["confidence"]] for d in dets],
            radius=28, blur=22, min_opacity=0.25,
            gradient={0.2:"#2c7fb8", 0.5:"#fdae61", 0.85:"#f03b20"}).add_to(m)
folium.PolyLine(route_latlon(route_plain), color="#1f78ff", weight=5, opacity=0.8,
                popup=f"shortest {d0:.0f} m").add_to(m)
folium.PolyLine(route_latlon(route_avoid), color="#2ca02c", weight=5, opacity=0.9,
                popup=f"low-pollen {d1:.0f} m").add_to(m)
folium.Marker([CAMPUS[0], CAMPUS[1]], popup="START: OIC", icon=folium.Icon(color="green")).add_to(m)
folium.Marker([STATION[0], STATION[1]], popup="JR Ibaraki", icon=folium.Icon(color="red")).add_to(m)
m.save("route.html")
print("\nwrote route.html  (blue = shortest, green = low-pollen)")
