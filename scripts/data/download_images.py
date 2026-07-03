
import argparse
import csv
import os
import time
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multimedia", default="multimedia.txt", help="GBIFのmultimedia.txtのパス")
    ap.add_argument("--out", required=True, help="保存先フォルダ 例: images/sugi")
    ap.add_argument("--limit", type=int, default=300, help="ダウンロードする最大枚数")
    ap.add_argument("--publisher", default="iNaturalist",
                    help="この publisher の行だけ採用。'any' で全件対象")
    ap.add_argument("--delay", type=float, default=0.3, help="各ダウンロード間の待機秒数")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # multimedia.txt を読む（タブ区切り、1行目がヘッダー）
    with open(args.multimedia, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # publisher でフィルタ
    if args.publisher.lower() != "any":
        rows = [r for r in rows if (r.get("publisher") or "").strip() == args.publisher]

    # 画像URLがある行だけ
    rows = [r for r in rows if (r.get("identifier") or "").strip()]
    print(f"対象レコード数: {len(rows)}（publisher={args.publisher}）")

    headers = {"User-Agent": "kafun-research/1.0 (academic use)"}

    saved = 0
    skipped = 0
    for r in rows:
        if saved >= args.limit:
            break
        url = r["identifier"].strip()
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            # 拡張子を判定（なければ jpg）
            ext = ".jpg"
            ctype = resp.headers.get("Content-Type", "")
            if "png" in ctype:
                ext = ".png"
            fname = os.path.join(args.out, f"{saved:04d}{ext}")
            with open(fname, "wb") as out:
                out.write(resp.content)
            saved += 1
            if saved % 25 == 0:
                print(f"  保存 {saved}/{args.limit} ...")
        except Exception as e:
            skipped += 1
            # リンク切れ等は飛ばす
        time.sleep(args.delay)

    print(f"完了: {saved} 枚保存, {skipped} 枚スキップ -> {args.out}")


if __name__ == "__main__":
    main()
