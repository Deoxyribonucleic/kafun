"""
End-to-end pipeline accuracy: runs every image in testset/<truth>/ through
plant-gate -> close/far gate -> BiRefNet -> species model, and scores the
final outcome (including correct rejections).
  python3 test_pipeline.py --model kafun_model.pth
testset layout: testset/{sugi,hinoki,other,notplant,far}/*.jpg
"""
import argparse, os, sys, glob, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, AutoModelForImageSegmentation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from model_utils import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--testdir", default="./testset")
ap.add_argument("--model", default="kafun_model.pth")
ap.add_argument("--thresh", type=float, default=0.70)
ap.add_argument("--margin", type=float, default=0.65)
ap.add_argument("--plant", type=float, default=0.50)
ap.add_argument("--res", type=int, default=1024)
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| model:", args.model)

# ---- load models once ----
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
bn = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device).float().eval()
ckpt = torch.load(args.model, map_location=device)
classes = ckpt["classes"]
spec = build_model(ckpt.get("arch","resnet50"), len(classes), head=ckpt.get("head","linear"), pretrained=False)
spec.load_state_dict(ckpt["state_dict"]); spec = spec.to(device).eval()

def text_feats(prompts):
    with torch.no_grad():
        tok = cproc(text=prompts, return_tensors="pt", padding=True).to(device)
        tf = clip.get_text_features(**tok)
        tf = tf if torch.is_tensor(tf) else tf.pooler_output
        return tf / tf.norm(dim=-1, keepdim=True)

pfeat = text_feats(["a photo of a plant","a photo of a tree","a photo of leaves or foliage",
                    "a photo of an animal","a photo of a person",
                    "a photo of an object or building","a photo of food"])
cffeat = text_feats(["a close-up photo of tree leaves","a macro photo of foliage and needles",
                     "a detailed shot of leaves filling the frame",
                     "a photo of a whole tree from a distance",
                     "a tree photographed far away with sky and background",
                     "a landscape photo containing a tree"])
btf = transforms.Compose([transforms.Resize((args.res,args.res)), transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
stf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def run(img):
    with torch.no_grad():
        pix = cproc(images=img, return_tensors="pt").to(device)
        ifeat = clip.get_image_features(**pix)
        ifeat = ifeat if torch.is_tensor(ifeat) else ifeat.pooler_output
        ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
        ps = (ifeat @ pfeat.T)[0]
        pc = torch.softmax(torch.stack([ps[:3].mean(), ps[3:].mean()])*100,0)
        if pc[0].item() <= args.plant: return "notplant"
        cs = (ifeat @ cffeat.T)[0]
        cc = torch.softmax(torch.stack([cs[:3].mean(), cs[3:].mean()])*100,0)
        if cc[0].item() <= args.margin: return "far"
        pred = bn(btf(img).unsqueeze(0).to(device).float())[-1].sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(img.size)
        cut = Image.new("RGB", img.size, (255,255,255)); cut.paste(img, mask=mask)
        probs = F.softmax(spec(stf(cut).unsqueeze(0).to(device)),1)[0]
        c, idx = probs.max(0)
        if c.item() < args.thresh: return "rejected_lowconf"
        return classes[idx]

# ---- run over testset ----
truths = [d for d in ["sugi","hinoki","other","notplant","far"]
          if os.path.isdir(os.path.join(args.testdir, d))]
total_ok = total = 0
print()
for truth in truths:
    files = [p for p in glob.glob(os.path.join(args.testdir, truth, "*"))
             if p.lower().endswith((".jpg",".jpeg",".png"))]
    outcomes = {}
    for p in files:
        try: img = Image.open(p).convert("RGB")
        except Exception: continue
        out = run(img)
        outcomes[out] = outcomes.get(out, 0) + 1
        # define "correct": species matches, OR notplant->notplant, OR far->far/rejected
        ok = (out == truth) or \
             (truth == "notplant" and out in ("notplant","rejected_lowconf")) or \
             (truth == "far" and out in ("far","rejected_lowconf","notplant"))
        total_ok += int(ok); total += 1
    print(f"[{truth:9s}] ({len(files)} imgs) -> {outcomes}")
print(f"\nEND-TO-END ACCURACY: {total_ok}/{total} = {total_ok/max(total,1):.2f}")
