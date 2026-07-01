"""
学習済みモデルの混同行列を出すスクリプト。
どのクラスをどのクラスと間違えているかを数値で確認する。
使い方:
    python3 evaluate.py --data_dir ./data --model kafun_model.pth
出力:
    - クラスごとの正解率
    - 混同行列（行=正解, 列=予測）
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def build_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"未知のアーキテクチャ: {arch}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", default="kafun_model.pth")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    classes = ckpt["classes"]
    n = len(classes)
    arch = ckpt.get("arch", "resnet18")
    model = build_model(arch, n)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), norm,
    ])
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), val_tf)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ImageFolder のクラス順が ckpt["classes"] と一致しているか確認
    if val_ds.classes != list(classes):
        print("⚠ 警告: val のクラス順がモデルのクラス順と違います")
        print("  val:", val_ds.classes)
        print("  model:", list(classes))

    cm = [[0] * n for _ in range(n)]
    with torch.no_grad():
        for x, y in val_ld:
            x = x.to(device)
            pred = model(x).argmax(1).cpu()
            for t, p in zip(y, pred):
                cm[t.item()][p.item()] += 1

    print("\nクラス:", classes)
    print("\n=== クラスごとの正解率（recall）===")
    for i, c in enumerate(classes):
        total = sum(cm[i])
        acc = cm[i][i] / total if total else 0
        print(f"  {c:8s}: {acc:.2f}  ({cm[i][i]}/{total})")

    # precision も出す（hinoki が他クラスを巻き込んでいないかの確認用）
    print("\n=== クラスごとの precision ===")
    for j, c in enumerate(classes):
        col = sum(cm[i][j] for i in range(n))
        prec = cm[j][j] / col if col else 0
        print(f"  {c:8s}: {prec:.2f}  ({cm[j][j]}/{col})")

    print("\n=== 混同行列 (行=正解, 列=予測) ===")
    header = "正解\\予測 | " + " ".join(f"{c[:6]:>6s}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = " ".join(f"{cm[i][j]:>6d}" for j in range(n))
        print(f"{c[:8]:>8s} | {row}")

if __name__ == "__main__":
    main()