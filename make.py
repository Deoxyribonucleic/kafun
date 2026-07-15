"""
Make a normalized confusion matrix (like the wolf/coyote/dog example)
for the Kafun classifier. Outputs a slide-ready PNG.

On your machine, run:  python3 make_confusion.py
Requires: torch, torchvision, matplotlib, seaborn, numpy, and your kafun_model.pth + data/
"""
import torch, numpy as np, os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- adjust these two paths to your setup ---
sys.path.insert(0, "src")            # where model_utils.py lives (or "." )
from model_utils import build_model

MODEL = "kafun_model.pth"
DATA  = "./data"
# presentation labels (consistent naming — English, no Japanese terms)
DISPLAY = {"cedar": "Cedar (Sugi)", "cypress": "Cypress (Hinoki)", "other": "Other",
           "sugi": "Cedar", "hinoki": "Cypress"}  # maps whatever your classes are called

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(MODEL, map_location=device)
classes = ckpt["classes"]; n = len(classes)
model = build_model(ckpt.get("arch","resnet50"), n, head=ckpt.get("head","linear"), pretrained=False)
model.load_state_dict(ckpt["state_dict"]); model = model.to(device).eval()

norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
ds = datasets.ImageFolder(os.path.join(DATA,"val"), tf)
ld = DataLoader(ds, batch_size=32, shuffle=False)

cm = np.zeros((n,n), int)
with torch.no_grad():
    for x,y in ld:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t,p in zip(y.numpy(), pred): cm[t][p]+=1

# row-normalize (each row sums to 1) -> same style as the example
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# nice display labels, reorder to cedar/cypress/other if present
labels = [DISPLAY.get(c, c.capitalize()) for c in classes]

plt.figure(figsize=(8,7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size":14,"weight":"bold"}, cbar=True,
            vmin=0, vmax=1, linewidths=0.5, linecolor="white", square=True)
plt.xlabel("Predicted label", fontsize=12)
plt.ylabel("True label", fontsize=12)
plt.title("Confusion Matrix (normalized)", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
print("saved confusion_matrix.png")
print("raw counts:\n", cm)