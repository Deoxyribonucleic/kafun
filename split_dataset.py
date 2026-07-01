"""
images/<クラス>/ の画像を train/val に分割するスクリプト。

使い方:
    python3 split_dataset.py --src images --dst data --val_ratio 0.2

動作:
    - images/ 直下の各サブフォルダ（=クラス）を読む
    - 各クラスの画像をシャッフルして train/val に分割
    - data/train/<クラス>/ と data/val/<クラス>/ にコピー
    - 元の images/ はそのまま残す（コピーなので安全）

例の入力:
    images/sugi/*.jpg
    images/hinoki/*.jpg
    images/other/*.jpg
例の出力:
    data/train/{sugi,hinoki,other}/...
    data/val/{sugi,hinoki,other}/...
"""
import argparse
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="images", help="クラス別フォルダがある場所")
    ap.add_argument("--dst", default="data", help="出力先（train/valを作る）")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="検証用に回す割合")
    ap.add_argument("--seed", type=int, default=42, help="分割の乱数シード")
    args = ap.parse_args()

    random.seed(args.seed)

    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    classes = [d for d in sorted(os.listdir(args.src))
               if os.path.isdir(os.path.join(args.src, d))]
    print("クラス:", classes)

    for cls in classes:
        src_dir = os.path.join(args.src, cls)
        files = [f for f in os.listdir(src_dir) if f.endswith(exts)]
        random.shuffle(files)

        n_val = int(len(files) * args.val_ratio)
        val_files = files[:n_val]
        train_files = files[n_val:]

        for split, flist in [("train", train_files), ("val", val_files)]:
            out_dir = os.path.join(args.dst, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for f in flist:
                shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))

        print(f"  {cls}: train={len(train_files)}  val={len(val_files)}")

    print(f"完了 -> {args.dst}/train, {args.dst}/val")


if __name__ == "__main__":
    main()