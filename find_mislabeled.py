"""
モデルが間違えた（予測≠正解ラベル）画像を集めるスクリプト。
ラベル修正の候補を効率よく見つけるためのもの。

使い方:
    python3 find_mislabeled.py --data_dir ./data --model kafun_model.pth --out review

動作:
    - data/train と data/val の全画像をモデルで判定
    - 予測が正解ラベルと食い違った画像を review/ にコピー
    - review/<正解>_AS_<予測>/ というフォルダ分けで、確信度つきファイル名で保存
      （例: review/hinoki_AS_other/ = ヒノキフォルダにあるのに other と判定された画像）
    - ファイル名の先頭が確信度。0.9 など高い確信度で間違えているものほど
      「誤ラベル or ゴミ」の可能性が高い（要チェック）

元の data/ は一切変更しません（コピーするだけ）。
"""
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def collect(split_dir, model, classes, device, tf, out_root):
    if not os.path.isdir(split_dir):
        return 0
    ds = datasets.ImageFolder(split_dir, tf)
    # ImageFolder の classes 並びと学習時の classes 並びが同じ前提
    n_moved = 0
    for path, label_idx in ds.samples:
        img = tf(datasets.folder.default_loader(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(img), dim=1)[0]
        conf, pred_idx = probs.max(0)
        pred_idx = pred_idx.item()
        if pred_idx != label_idx:
            true_name = ds.classes[label_idx]
            pred_name = ds.classes[pred_idx]
            sub = os.path.join(out_root, f"{true_name}_AS_{pred_name}")
            os.makedirs(sub, exist_ok=True)
            base = os.path.basename(path)
            # ファイル名先頭に確信度を付ける（高いほど怪しい）
            newname = f"{conf.item():.2f}_{base}"
            shutil.copy(path, os.path.join(sub, newname))
            n_moved += 1
    return n_moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", default="kafun_model.pth")
    ap.add_argument("--out", default="review")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.model, map_location=device)
    classes = ckpt["classes"]
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), norm,
    ])

    total = 0
    for split in ["train", "val"]:
        d = os.path.join(args.data_dir, split)
        moved = collect(d, model, classes, device, tf, args.out)
        print(f"{split}: {moved} 枚の食い違いを {args.out}/ にコピー")
        total += moved

    print(f"\n合計 {total} 枚。{args.out}/ の各フォルダを開いて確認してください。")
    print("フォルダ名 <正解>_AS_<予測> = 「正解フォルダにあるが予測と食い違った画像」")
    print("ファイル名先頭の数字が確信度。高い（0.8以上）ほど誤ラベル/ゴミの可能性大。")


if __name__ == "__main__":
    main()