import json, folium
from folium.plugins import HeatMap

with open("detections.json") as f:
    dets = json.load(f)

CENTER = [34.8085, 135.5605]

heat = [[d["lat"], d["lon"], d["confidence"]]
        for d in dets if d["species"] in ("sugi", "hinoki")]
others = [d for d in dets if d["species"] == "other"]

m = folium.Map(location=CENTER, zoom_start=17, tiles="CartoDB positron")

HeatMap(heat, radius=22, blur=18, min_opacity=0.3, max_zoom=18,
        gradient={0.2: "#2c7fb8", 0.4: "#7fcdbb", 0.6: "#fdae61", 0.85: "#f03b20"}
        ).add_to(m)

for d in others:
    folium.CircleMarker([d["lat"], d["lon"]], radius=2,
                        color="#888888", fill=True, fill_opacity=0.5,
                        popup="other (non-allergenic)").add_to(m)

legend = (
    '<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;'
    ' background: white; padding: 10px 14px; border: 1px solid #999;'
    ' border-radius: 6px; font-family: sans-serif; font-size: 13px;">'
    '<b>Pollen risk (sugi + hinoki)</b><br>'
    '<span style="color:#2c7fb8">&#9632;</span> low &nbsp;'
    '<span style="color:#fdae61">&#9632;</span> med &nbsp;'
    '<span style="color:#f03b20">&#9632;</span> high<br>'
    '<span style="color:#888">&#9679;</span> other (not counted)'
    '</div>'
)
m.get_root().html.add_child(folium.Element(legend))

m.save("heatmap.html")
print(f"wrote heatmap.html  | {len(heat)} allergenic points, {len(others)} other")
