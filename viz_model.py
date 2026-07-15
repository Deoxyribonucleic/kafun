import torch, numpy as np, os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
sys.path.insert(0, "src")  # adjust if model_utils is elsewhere
from model_utils import build_model

MODEL = "kafun_model.pth"
DATA  = "./data"
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

# ---- 1. confusion matrix heatmap ----
fig, ax = plt.subplots(figsize=(5.5,4.8))
im = ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(classes); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (validation)")
for i in range(n):
    for j in range(n):
        ax.text(j,i,cm[i,j],ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=13, fontweight="bold")
fig.colorbar(im, fraction=0.046)
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

# ---- 2. per-class recall / precision bars ----
recall = [cm[i,i]/cm[i].sum() if cm[i].sum() else 0 for i in range(n)]
prec   = [cm[i,i]/cm[:,i].sum() if cm[:,i].sum() else 0 for i in range(n)]
x = np.arange(n); w=0.35
fig, ax = plt.subplots(figsize=(6,4.5))
ax.bar(x-w/2, recall, w, label="Recall", color="#2ca02c")
ax.bar(x+w/2, prec,   w, label="Precision", color="#e0a526")
ax.set_xticks(x); ax.set_xticklabels(classes); ax.set_ylim(0,1.05)
ax.set_ylabel("Score"); ax.set_title("Per-class Recall & Precision")
for i in range(n):
    ax.text(i-w/2, recall[i]+0.02, f"{recall[i]:.2f}", ha="center", fontsize=9)
    ax.text(i+w/2, prec[i]+0.02,   f"{prec[i]:.2f}",   ha="center", fontsize=9)
ax.legend(); plt.tight_layout(); plt.savefig("outputs/metrics_bars.png", dpi=150); plt.close()

acc = np.trace(cm)/cm.sum()
print(f"overall accuracy: {acc:.3f}")
print("saved -> outputs/confusion_matrix.png, outputs/metrics_bars.png")
