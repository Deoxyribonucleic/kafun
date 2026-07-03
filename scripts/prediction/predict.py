"""
学習済みモデルで1枚の写真を判定するスクリプト。
確信度がしきい値未満なら「該当なし」を返す（2クラスでも未知入力を弾くため）。
使い方:
    python3 predict.py <画像ファイル> [--thresh 0.7]
例:
    python3 predict.py test.jpg
    python3 predict.py test.jpg --thresh 0.8
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from model_utils import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="判定する画像ファイルのパス")
    ap.add_argument("--model", default="kafun_model.pth")
    ap.add_argument("--thresh", type=float, default=0.7,
                    help="この確信度未満なら『該当なし』")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt["classes"]
    arch = ckpt.get("arch", "resnet18")  # 古いチェックポイント用にフォールバック
    model = build_model(arch, len(classes), head=ckpt.get("head", "linear"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = tf(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1)[0]
    conf, idx = probs.max(0)

    print("確信度:", {classes[i]: round(probs[i].item(), 2) for i in range(len(classes))})
    if conf.item() < args.thresh:
        print(f"=> 該当なし (最大確信度 {conf.item():.2f} < しきい値 {args.thresh})")
    else:
        print(f"=> {classes[idx]} (確信度 {conf.item():.2f})")

if __name__ == "__main__":
    main()