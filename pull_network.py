import osmnx as ox
import folium, math

CAMPUS = (34.8085, 135.5605)
STATION = (34.8157, 135.5630)

mid = ((CAMPUS[0]+STATION[0])/2, (CAMPUS[1]+STATION[1])/2)
def m(a,b):
    dy=(a[0]-b[0])*111320; dx=(a[1]-b[1])*111320*math.cos(math.radians(a[0]))
    return (dx*dx+dy*dy)**0.5
radius = int(max(m(mid,CAMPUS), m(mid,STATION)) + 400)
print(f"pulling walk network: center {mid}, radius {radius} m")

G = ox.graph_from_point(mid, dist=radius, network_type="walk")
print(f"nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")
ox.save_graphml(G, "oic_walk.graphml")
print("saved -> oic_walk.graphml")

fmap = folium.Map(location=mid, zoom_start=16, tiles="CartoDB positron")
for u, v, data in G.edges(data=True):
    ys = [G.nodes[u]["y"], G.nodes[v]["y"]]
    xs = [G.nodes[u]["x"], G.nodes[v]["x"]]
    folium.PolyLine(list(zip(ys, xs)), color="#3388ff", weight=1.5, opacity=0.6).add_to(fmap)
folium.Marker(CAMPUS, popup="START: OIC campus", icon=folium.Icon(color="green")).add_to(fmap)
folium.Marker(STATION, popup="END: JR Ibaraki", icon=folium.Icon(color="red")).add_to(fmap)
fmap.save("network_preview.html")
print("wrote network_preview.html")
