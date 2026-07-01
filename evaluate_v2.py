import argparse, os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_utils import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data_cb")
    ap.add_argument("--model", default="kafun_cb.pth")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.model, map_location=device)
    classes = ckpt["classes"]; n = len(classes)
    model = build_model(ckpt.get("arch","resnet50"), n,
                        head=ckpt.get("head","linear"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"]); model = model.to(device).eval()

    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), norm])
    ds = datasets.ImageFolder(os.path.join(args.data_dir,"val"), tf)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    if ds.classes != list(classes):
        print("WARNING class order: val", ds.classes, "vs model", list(classes))

    cm = [[0]*n for _ in range(n)]
    with torch.no_grad():
        for x,y in ld:
            pred = model(x.to(device)).argmax(1).cpu()
            for t,p in zip(y,pred): cm[t.item()][p.item()] += 1

    print("\nclasses:", classes)
    print("\n=== recall ===")
    for i,c in enumerate(classes):
        tot=sum(cm[i]); print(f"  {c:8s}: {cm[i][i]/tot if tot else 0:.2f}  ({cm[i][i]}/{tot})")
    print("\n=== precision ===")
    for j,c in enumerate(classes):
        col=sum(cm[i][j] for i in range(n)); print(f"  {c:8s}: {cm[j][j]/col if col else 0:.2f}  ({cm[j][j]}/{col})")
    print("\n=== confusion (row=true, col=pred) ===")
    print("true\\pred | " + " ".join(f"{c[:6]:>6s}" for c in classes))
    for i,c in enumerate(classes):
        print(f"{c[:8]:>8s} | " + " ".join(f"{cm[i][j]:>6d}" for j in range(n)))

if __name__ == "__main__":
    main()
