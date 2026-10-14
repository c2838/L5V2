"""執行方式：.venv/bin/python run.py --dataset all"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 圖表 cache 留在專案環境內。
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from datasets import load_dataset
from analysis import analyze
from export import build_site, export_result


def main():
    parser = argparse.ArgumentParser(description="AIoT 多元線性回歸：Python 分析與靜態網頁")
    parser.add_argument("--dataset", choices=["all", "startup", "boston", "custom"], default="all")
    parser.add_argument("--csv", help="自訂的公開 HTTPS CSV 網址或本機路徑")
    parser.add_argument("--target", help="自訂資料的數值目標欄位")
    parser.add_argument("--refresh", action="store_true", help="重新下載預設資料")
    parser.add_argument("--site-only", action="store_true", help="由既有結果重新產生網頁")
    args = parser.parse_args()
    if args.csv and args.dataset != "custom":
        parser.error("--csv 請搭配 --dataset custom，避免覆寫預設題目的資料。")
    if args.dataset == "custom" and (not args.csv or not args.target) and not args.site_only:
        parser.error("自訂資料需要 --csv 與 --target。")
    try:
        if not args.site_only:
            for name in (["startup", "boston"] if args.dataset == "all" else [args.dataset]):
                print(f"分析 {name} …", flush=True)
                frame, metadata = load_dataset(ROOT, name, args.csv, args.target, args.refresh)
                result, model, verification = analyze(frame, metadata, progress=lambda s: print(s, flush=True))
                export_result(ROOT, result, model, verification)
                print(f"  已輸出 outputs/{name}", flush=True)
        build_site(ROOT)
        print("靜態成果已建立：docs/index.html", flush=True)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"分析未完成：{exc}\n")


if __name__ == "__main__":
    main()
