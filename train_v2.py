"""
Training with the paper's techniques: dropout head, optional frozen backbone, arch choice.
  python3 train_v2.py --data_dir ./data_cb --arch resnet50 --head dropout --out kafun_cb.pth
  add --freeze to train only the head (backbone frozen)
"""
import argparse, os, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_utils import build_model

def build_loaders(data_dir, batch_size):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), norm,
    ])
    tr = datasets.ImageFolder(os.path.join(data_dir,"train"), train_tf)
    va = datasets.ImageFolder(os.path.join(data_dir,"val"), val_tf)
    return (DataLoader(tr,batch_size,shuffle=True,num_workers=4),
            DataLoader(va,batch_size,shuffle=False,num_workers=4), tr.classes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data_cb")
    ap.add_argument("--arch", default="resnet50", choices=["resnet18","resnet50"])
    ap.add_argument("--head", default="dropout", choices=["linear","dropout"])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--out", default="kafun_cb.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| arch:", args.arch, "| head:", args.head, "| freeze:", args.freeze)

    train_ld, val_ld, classes = build_loaders(args.data_dir, args.batch_size)
    print("classes:", classes)

    model = build_model(args.arch, len(classes), head=args.head,
                        freeze=args.freeze, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    for ep in range(args.epochs):
        model.train(); rl = 0.0
        for x,y in train_ld:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y); loss.backward(); optimizer.step()
            rl += loss.item()
        scheduler.step()
        model.eval(); c=t=0
        with torch.no_grad():
            for x,y in val_ld:
                x,y = x.to(device), y.to(device)
                c += (model(x).argmax(1)==y).sum().item(); t += y.size(0)
        acc = c/max(t,1)
        print(f"epoch {ep+1}/{args.epochs}  loss={rl/len(train_ld):.3f}  val_acc={acc:.3f}")
        if acc > best:
            best = acc
            torch.save({"state_dict":model.state_dict(), "classes":classes,
                        "arch":args.arch, "head":args.head}, args.out)
            print(f"  saved best -> {args.out} (acc={acc:.3f})")
    print("done. best val_acc =", round(best,3))

if __name__ == "__main__":
    main()
