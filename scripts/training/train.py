"""
Kafun classifier - transfer learning (ResNet50 版)
花粉植物の画像分類モデルを学習するスクリプト。

ResNet18版からの変更点:
    - 土台モデルを ResNet50 に変更（表現力アップ）
    - 既定バッチサイズを 16 に（ResNet50 は重いのでメモリ対策）
    - weight decay を追加（大きいモデルの過学習を抑制）

使い方:
    python3 train.py --data_dir ./data --epochs 25 --batch_size 16

メモリ不足（CUDA out of memory）が出たら --batch_size 8 に下げる。
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from model_utils import build_model


def build_loaders(data_dir, batch_size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_ld, val_ld, train_ds.classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--out", default="kafun_model.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cpu":
        print("警告: GPU が検出されませんでした。CPU だと非常に遅くなります。")

    train_ld, val_ld, classes = build_loaders(args.data_dir, args.batch_size)
    print("classes:", classes)

    # 学習済み ResNet50 を読み込み、最後の層だけ自分のクラス数に差し替える
    model = build_model("resnet50", len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # --- 学習 ---
        model.train()
        running_loss = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # --- 検証 ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch + 1}/{args.epochs}  loss={running_loss/len(train_ld):.3f}  val_acc={acc:.3f}  lr={cur_lr:.2e}")

        if acc > best_acc:
            best_acc = acc
            # arch も保存しておく（評価/判定スクリプトがどのモデルか分かるように）
            torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": "resnet50"}, args.out)
            print(f"  saved best -> {args.out} (acc={acc:.3f})")

    print("done. best val_acc =", round(best_acc, 3))


if __name__ == "__main__":
    main()