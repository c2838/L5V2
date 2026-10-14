"""資料載入、來源紀錄與輸入檢查；不依賴 Notebook 狀態。"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.io import arff

PRESETS = {
    "startup": {
        "title": "50 Startups", "subtitle": "新創公司的獲利，能由支出預測嗎？",
        "target": "Profit", "unit": "資料集貨幣單位",
        "url": "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv",
    },
    "boston": {
        "title": "Boston Housing", "subtitle": "哪些地區特徵有助於預測房價？",
        "target": "MEDV", "unit": "千美元",
        "url": "https://www.openml.org/data/v1/download/52643/boston.arff",
        "reference": "https://www.openml.org/search?type=data&id=531",
        "version": "OpenML data_id=531, version=1",
    },
}


def download(url: str) -> bytes:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("資料網址必須是沒有帳密的公開 HTTPS 網址。")
    try:
        with urlopen(Request(url, headers={"User-Agent": "AIoT-coursework/1.0"}), timeout=30) as response:
            content = response.read(10_000_001)
    except Exception as exc:
        raise ValueError(f"資料下載失敗（{type(exc).__name__}）；請確認網路或提供本機 CSV。") from exc
    if len(content) > 10_000_000:
        raise ValueError("資料超過本次作業的 10 MB 限制。")
    return content


def load_dataset(root: Path, name: str, csv_source: str | None = None,
                 target: str | None = None, refresh: bool = False):
    config = dict(PRESETS.get(name, {"title": "Custom Dataset", "subtitle": "使用相同 workflow 分析自訂資料", "unit": "目標欄位單位"}))
    target = target or config.get("target")
    if not target:
        raise ValueError("自訂資料必須透過 --target 指定要預測的欄位。")
    config["target"] = target
    folder = root / "data" / name
    folder.mkdir(parents=True, exist_ok=True)
    if csv_source:
        if urlsplit(csv_source).scheme:
            content, source = download(csv_source), csv_source
        else:
            path = Path(csv_source).expanduser().resolve()
            if not path.is_file():
                raise ValueError("找不到指定的 CSV 檔案。")
            if path.stat().st_size > 10_000_000:
                raise ValueError("資料超過本次作業的 10 MB 限制。")
            content, source = path.read_bytes(), "local CSV: " + path.name
        frame = pd.read_csv(io.BytesIO(content))
        (folder / "source.csv").write_bytes(content)
    else:
        extension = "arff" if name == "boston" else "csv"
        raw_path = folder / f"source.{extension}"
        if refresh or not raw_path.exists():
            raw_path.write_bytes(download(config["url"]))
        content, source = raw_path.read_bytes(), config["url"]
        if extension == "arff":
            records, _ = arff.loadarff(io.StringIO(content.decode("utf-8")))
            frame = pd.DataFrame(records)
            # OpenML 531 的 CHAS/RAD 在 ARFF 宣告為 nominal，值以 bytes 載入。
            # 本題原始 13 個欄位都是數值（含二元與離散值），保留原始表示。
            for col in frame:
                frame[col] = frame[col].map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)
                frame[col] = pd.to_numeric(frame[col], errors="raise")
        else:
            frame = pd.read_csv(io.BytesIO(content))
    if frame.columns.duplicated().any():
        raise ValueError("CSV 含有重複欄位名稱。")
    if target not in frame:
        raise ValueError(f"找不到目標欄位 {target}。可用欄位：{', '.join(map(str, frame.columns))}")
    target_numeric = pd.to_numeric(frame[target], errors="coerce")
    invalid_target = frame[target].notna() & target_numeric.isna()
    if invalid_target.any() or np.isinf(target_numeric).any():
        raise ValueError("回歸目標必須是有限數值；請檢查文字或無限值。")
    dropped_target = int(target_numeric.isna().sum())
    frame[target] = target_numeric
    frame = frame.loc[target_numeric.notna()].copy()
    if len(frame) < 25:
        raise ValueError("有效資料至少需要 25 筆，才能切分測試集並做交叉驗證。")
    if frame[target].nunique() < 2:
        raise ValueError("目標值全部相同，無法進行有意義的回歸評估。")
    features = frame.drop(columns=target)
    if features.empty or features.shape[1] > 20:
        raise ValueError("本次 workflow 支援 1～20 個原始特徵欄位。")
    if features.isna().all().any():
        raise ValueError("有特徵欄位完全沒有資料，請先移除或補齊。")
    for col in features.select_dtypes(include="number"):
        if np.isinf(features[col].to_numpy(dtype=float)).any():
            raise ValueError(f"特徵 {col} 含無限值。")
    for col in features.select_dtypes(exclude="number"):
        if features[col].nunique() > 12:
            raise ValueError(f"文字特徵 {col} 超過 12 種值；請排除識別碼或先整理類別。")
        # sklearn 的類別前處理採用統一的 object/np.nan 表示。
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    metadata = {
        "name": name, **config, "source": source,
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(frame), "raw_feature_count": len(features.columns),
        "missing_values": {c: int(frame[c].isna().sum()) for c in frame.columns},
        "dropped_missing_targets": dropped_target,
        "duplicate_rows": int(frame.duplicated().sum()),
        "columns": list(frame.columns),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(folder / "cleaned.csv", index=False)
    return frame, metadata
