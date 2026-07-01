"""
Classify a photo and, if it's sugi/hinoki with a location, add it to detections.json
(the crowdsource data that feeds the heatmap).
  python3 enroll_detection.py photo.jpg                 # uses GPS from photo EXIF
  python3 enroll_detection.py photo.jpg --lat 34.809 --lon 135.560   # manual location
"""
import argparse, json, os, datetime, torch
import torch.nn.functional as F
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, AutoModelForImageSegmentation
from model_utils import build_model

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--model", default="kafun_model.pth")
ap.add_argument("--lat", type=float, default=None)
ap.add_argument("--lon", type=float, default=None)
ap.add_argument("--thresh", type=float, default=0.70)
ap.add_argument("--out", default="detections.json")
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- read GPS from EXIF if not given manually ----
def exif_gps(path):
    try:
        img = Image.open(path); exif = img._getexif()
        if not exif: return None
        gps = None
        for tag, val in exif.items():
            if TAGS.get(tag) == "GPSInfo":
                gps = {GPSTAGS.get(t, t): v for t, v in val.items()}
        if not gps or "GPSLatitude" not in gps: return None
        def dms(v): return v[0] + v[1]/60 + v[2]/3600
        lat = dms(gps["GPSLatitude"]);  lon = dms(gps["GPSLongitude"])
        if gps.get("GPSLatitudeRef") == "S":  lat = -lat
        if gps.get("GPSLongitudeRef") == "W": lon = -lon
        return float(lat), float(lon)
    except Exception:
        return None

if args.lat is not None and args.lon is not None:
    loc = (args.lat, args.lon)
else:
    loc = exif_gps(args.image)

if loc is None:
    print("=> no location (photo has no GPS and none given via --lat/--lon). Not added.")
    raise SystemExit

# ---- classify (reuse the gated pipeline briefly) ----
img = Image.open(args.image).convert("RGB")
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def tfeat(prompts):
    with torch.no_grad():
        t = cproc(text=prompts, return_tensors="pt", padding=True).to(device)
        f = clip.get_text_features(**t); f = f if torch.is_tensor(f) else f.pooler_output
        return f / f.norm(dim=-1, keepdim=True)
with torch.no_grad():
    pix = cproc(images=img, return_tensors="pt").to(device)
    ifeat = clip.get_image_features(**pix); ifeat = ifeat if torch.is_tensor(ifeat) else ifeat.pooler_output
    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
pf = tfeat(["a photo of a plant","a photo of a tree","a photo of leaves or foliage",
            "a photo of an animal","a photo of a person","a photo of an object","a photo of food"])
ps = (ifeat @ pf.T)[0]
if torch.softmax(torch.stack([ps[:3].mean(), ps[3:].mean()])*100,0)[0].item() <= 0.5:
    print("=> not a plant. Not added."); raise SystemExit
del clip; torch.cuda.empty_cache()

bn = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device).float().eval()
btf = transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
with torch.no_grad():
    pred = bn(btf(img).unsqueeze(0).to(device).float())[-1].sigmoid().cpu()[0].squeeze()
mask = transforms.ToPILImage()(pred).resize(img.size)
cut = Image.new("RGB", img.size, (255,255,255)); cut.paste(img, mask=mask)
del bn; torch.cuda.empty_cache()

ckpt = torch.load(args.model, map_location=device)
classes = ckpt["classes"]
model = build_model(ckpt.get("arch","resnet50"), len(classes), head=ckpt.get("head","linear"), pretrained=False)
model.load_state_dict(ckpt["state_dict"]); model = model.to(device).eval()
stf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
with torch.no_grad():
    probs = F.softmax(model(stf(cut).unsqueeze(0).to(device)),1)[0]
conf, idx = probs.max(0)
species = classes[idx]; conf = conf.item()
print(f"classified: {species} ({conf:.2f}) at {loc[0]:.6f}, {loc[1]:.6f}")

# ---- only allergenic conifers with confidence get added ----
if conf < args.thresh:
    print(f"=> low confidence (<{args.thresh}). Not added."); raise SystemExit
if species not in ("sugi", "hinoki"):
    print(f"=> '{species}' is not an allergenic conifer. Not added to pollen map."); raise SystemExit

det = {"lat": round(loc[0],6), "lon": round(loc[1],6), "species": species,
       "confidence": round(conf,2), "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
data = json.load(open(args.out)) if os.path.exists(args.out) else []
data.append(det)
json.dump(data, open(args.out,"w"), indent=2)
print(f"=> ADDED to {args.out}  (total detections: {len(data)})")
