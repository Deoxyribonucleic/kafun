import argparse, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, AutoModelForImageSegmentation
from model_utils import build_model

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--model", default="kafun_cb.pth")
ap.add_argument("--thresh", type=float, default=0.70)
ap.add_argument("--margin", type=float, default=0.65)   # close/far cutoff
ap.add_argument("--plant", type=float, default=0.50)    # plant/not-plant cutoff
ap.add_argument("--res", type=int, default=1024)
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
img = Image.open(args.image).convert("RGB")

# ---- CLIP setup ----
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def text_feats(prompts):
    with torch.no_grad():
        tok = cproc(text=prompts, return_tensors="pt", padding=True).to(device)
        tf = clip.get_text_features(**tok)
        tf = tf if torch.is_tensor(tf) else tf.pooler_output
        return tf / tf.norm(dim=-1, keepdim=True)

# image embedding (computed once, reused by both gates)
with torch.no_grad():
    pix = cproc(images=img, return_tensors="pt").to(device)
    ifeat = clip.get_image_features(**pix)
    ifeat = ifeat if torch.is_tensor(ifeat) else ifeat.pooler_output
    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)

# ---- Gate 1: is this a plant at all? ----
plant_prompts = ["a photo of a plant", "a photo of a tree", "a photo of leaves or foliage"]
notplant_prompts = ["a photo of an animal", "a photo of a person",
                    "a photo of an object or building", "a photo of food"]
pfeat = text_feats(plant_prompts + notplant_prompts)
n_plant = len(plant_prompts)
psims = (ifeat @ pfeat.T)[0]
pconf = torch.softmax(torch.stack([psims[:n_plant].mean(), psims[n_plant:].mean()]) * 100, 0)
if pconf[0].item() <= args.plant:
    print(f"=> 植物ではありません / not a plant (plant score {pconf[0]:.2f}). Rejected.")
    raise SystemExit

# ---- Gate 2: close-up or too far? ----
cf_prompts = ["a close-up photo of tree leaves", "a macro photo of foliage and needles",
              "a detailed shot of leaves filling the frame",
              "a photo of a whole tree from a distance",
              "a tree photographed far away with sky and background",
              "a landscape photo containing a tree"]
cffeat = text_feats(cf_prompts)
sims = (ifeat @ cffeat.T)[0]
conf = torch.softmax(torch.stack([sims[:3].mean(), sims[3:].mean()]) * 100, 0)
if conf[0].item() <= args.margin:
    print(f"=> 遠すぎます / too far or unclear (close={conf[0]:.2f}). Please retake a CLOSE-UP of the leaves.")
    raise SystemExit
del clip; torch.cuda.empty_cache()

# ---- BiRefNet bg-remove ----
bn = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device).float().eval()
btf = transforms.Compose([transforms.Resize((args.res, args.res)), transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
with torch.no_grad():
    pred = bn(btf(img).unsqueeze(0).to(device).float())[-1].sigmoid().cpu()[0].squeeze()
mask = transforms.ToPILImage()(pred).resize(img.size)
cut = Image.new("RGB", img.size, (255,255,255)); cut.paste(img, mask=mask)
del bn; torch.cuda.empty_cache()

# ---- species model ----
ckpt = torch.load(args.model, map_location=device)
classes = ckpt["classes"]
model = build_model(ckpt.get("arch","resnet50"), len(classes), head=ckpt.get("head","linear"), pretrained=False)
model.load_state_dict(ckpt["state_dict"]); model = model.to(device).eval()
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
with torch.no_grad():
    probs = F.softmax(model(tf(cut).unsqueeze(0).to(device)), 1)[0]
c, idx = probs.max(0)
print("probs:", {classes[i]: round(probs[i].item(), 2) for i in range(len(classes))})
if c.item() < args.thresh:
    print(f"=> 該当なし / no confident match ({c.item():.2f} < {args.thresh})")
else:
    print(f"=> {classes[idx]} ({c.item():.2f})")

