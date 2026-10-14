"""匯出真實運算結果、圖表、模型與可獨立開啟的靜態網站。"""
from __future__ import annotations

import gzip
import json
import pickle
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def chart_style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#34463f", "text.color": "#24372e", "axes.edgecolor": "#d1d9d0"})


def export_result(root, result, model, verification):
    name = result["metadata"]["name"]
    folder = root / "outputs" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    records = [{"method": r["method"], "features": " | ".join(r["features"]), "k": r["k"], "alpha": r["alpha"], "l1_ratio": r["l1_ratio"],
                "CV_RMSE": r["cv_rmse"], "CV_RMSE_std": r["cv_rmse_std"],
                "test_adjusted_R2": r["statistics"]["test_adjusted_r2"], "training_F": r["statistics"]["f_stat"], "training_F_p": r["statistics"]["f_p_value"],
                **{f"train_{k}": v for k, v in r["train"].items()}, **{f"test_{k}": v for k, v in r["test"].items()}}
               for r in result["rows"]]
    pd.DataFrame(records).to_csv(folder / "comparison.csv", index=False)
    best = next(r for r in result["rows"] if r["key"] == result["best_key"])
    pd.DataFrame(best["predictions"]).to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(best["statistics"]["coefficients"]).to_csv(folder / "coefficients.csv", index=False)
    with gzip.open(folder / "model.pkl.gz", "wb") as file:
        pickle.dump(model, file)
    with gzip.open(folder / "model.pkl.gz", "rb") as file:
        loaded_model = pickle.load(file)
    np.testing.assert_allclose(loaded_model.predict(verification["X_test"]), model.predict(verification["X_test"]), rtol=1e-12, atol=1e-9)
    assert set(verification["train_indices"]).isdisjoint(verification["test_indices"])
    predicted = np.array([p["predicted"] for p in best["predictions"]])
    actual = np.array([p["actual"] for p in best["predictions"]])
    np.testing.assert_allclose(np.mean((predicted-actual)**2), best["test"]["mse"], rtol=1e-12)
    (folder / "verification.json").write_text(json.dumps({
        "scope": "real dataset / local sklearn execution",
        "train_test_disjoint": True, "model_reload_predictions_match": True, "exported_mse_recomputed": True,
        "train_indices": verification["train_indices"], "test_indices": verification["test_indices"],
    }, indent=2), encoding="utf-8")
    chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, color, style in [("lasso", "#6258c7", "-"), ("forward", "#258960", "--"), ("rfe", "#dc9744", ":"), ("elasticnet", "#cf6088", "-."), ("exhaustive", "#3b90b3", (0, (3, 2, 1, 2)))]:
        rows = [r for r in result["rows"] if r["method"] == method]
        if not rows:
            continue
        for axis, metric in zip(axes, ["rmse", "r2"]):
            axis.plot([r["k"] for r in rows], [r["test"][metric] for r in rows], marker="o", linestyle=style, color=color, label=method, markersize=3)
            axis.set(xlabel="Number of encoded features", ylabel=f"Test {metric.upper()}")
            axis.legend()
    fig.tight_layout(); fig.savefig(folder / "feature_comparison.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(actual, predicted, color="#28745b", alpha=.75)
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[0].plot([low, high], [low, high], "--", color="#c58c32")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="Actual vs. predicted")
    axes[1].scatter(predicted, actual-predicted, color="#28745b", alpha=.75)
    axes[1].axhline(0, linestyle="--", color="#c58c32")
    axes[1].set(xlabel="Predicted", ylabel="Actual - predicted", title="Residual diagnostic")
    fig.tight_layout(); fig.savefig(folder / "prediction_diagnostics.png", dpi=170); plt.close(fig)
    curve = result["learning_curve"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(curve["sizes"], curve["train_rmse"], "o-", color="#28745b", label="Training")
    axis.plot(curve["sizes"], curve["validation_rmse"], "o-", color="#c58c32", label="Cross-validation")
    axis.set(xlabel="Training sample count", ylabel="RMSE", title="Learning curve: all-feature baseline")
    axis.legend(); fig.tight_layout(); fig.savefig(folder / "learning_curve.png", dpi=170); plt.close(fig)
    baseline = next(r for r in result["rows"] if r["method"] == "baseline")
    report = f"""# {result['metadata']['title']} 分析結果

## 問題與資料
預測目標：`{result['metadata']['target']}`；單位：{result['metadata']['unit']}。
資料：{result['metadata']['rows']} 筆、{result['metadata']['raw_feature_count']} 個原始特徵；編碼後 {result['encoded_feature_count']} 個特徵。
來源：{result['metadata']['source']}
SHA-256：`{result['metadata']['sha256']}`

## 方法
固定 random_state={result['seed']}，80% 訓練、20% 最終測試。
比較 Lasso、Forward SFS、RFE、ElasticNet、Exhaustive Search 五種特徵選擇方法，另保留全特徵 OLS 基準。
訓練集 5-fold CV 選擇方法、特徵數與參數；Forward Selection 的每一步使用內層 CV。
缺值填補、類別編碼與標準化只學習當次訓練 fold。RFE 逐次移除標準化係數較小的特徵。
Lasso / ElasticNet 依標準化係數絕對值排序；固定 k 可能納入零係數尾端。五種方法的最終預測皆由選定子集合重新訓練 LinearRegression。
Exhaustive Search 實際枚舉 {result['exhaustive_search']['subsets_evaluated']} 個非空子集合，用 training CV 選擇各 k 的最佳子集合；超過 13 個編碼後特徵時停用此方法。
CV 分數是用於選擇的分數，尤其大量子集合搜尋可能造成選擇偏差；獨立 test set 保留到最後評估。

## 實際結果
CV 選定：**{best['method_label']}，{best['k']} 個特徵**。
特徵：{', '.join(best['features'])}。
CV RMSE：{best['cv_rmse']:.6f}（fold 標準差 {best['cv_rmse_std']:.6f}）。
Test MAE：{best['test']['mae']:.6f}；MSE：{best['test']['mse']:.6f}；RMSE：{best['test']['rmse']:.6f}；R²：{best['test']['r2']:.6f}。
Test Adjusted R²：{best['statistics']['test_adjusted_r2']}；Training OLS F-statistic：{best['statistics']['f_stat']}；F p-value：{best['statistics']['f_p_value']}。
全特徵基準 Test RMSE：{baseline['test']['rmse']:.6f}。
本次選定模型的測試誤差{'低於' if best['test']['rmse'] < baseline['test']['rmse'] else '不低於'}全特徵基準；這是這次保留測試集上的觀察，不保證其他切分也相同。

## 解讀與限制
MAE 是平均絕對誤差，與目標同單位；MSE 的單位是目標的平方；RMSE 與目標同單位；R² 不是分類準確率，且可能為負。
不同 k 的 test 指標只供描述，最佳模型標籤不使用 test 分數挑選。
Learning curve 為全特徵基準模型，使用 training/validation 誤差診斷；單一曲線或單次測試差距不能直接證明過擬合。
特徵係數描述本資料中的關聯，不代表因果關係。兩題的 RMSE 單位不同，不直接比較其大小。
係數標準誤、t/p、95% CI、F 與 VIF 由 training OLS 計算，需線性模型的統計假設；此推論以選定欄位為條件，未修正特徵選擇造成的影響，因此只作探索。Adjusted R² 則由 test R²、test 筆數與 k 計算。
本作業是歷史資料上的教學分析；小樣本與單次 80/20 切分限制了結果的穩定性。
Boston 使用原始歷史欄位供課堂重現，其中部分欄位具有時代與偏差限制；結果不作實際房價決策。

## 本機驗證
已核對 train/test 索引不重疊、模型重新載入預測一致，並由匯出預測重新計算 MSE。
這些驗證為本機實際資料運算，不代表已發布到 GitHub Pages。
"""
    (folder / "report.md").write_text(report, encoding="utf-8")


def build_site(root):
    results = []
    for name in ("startup", "boston", "custom"):
        path = root / "outputs" / name / "results.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        raise ValueError("沒有分析結果，請先執行分析。")
    target = root / "docs"
    target.mkdir(exist_ok=True)
    for file in (root / "web").iterdir():
        if file.is_file():
            shutil.copy2(file, target / file.name)
    # 用 JS 資料檔支援雙擊 index.html，不必依賴 fetch/file://。
    data = json.dumps(results, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (target / "data.js").write_text("window.AIOT_RESULTS = " + data + ";\n", encoding="utf-8")
    for result in results:
        name = result["metadata"]["name"]
        destination = target / "downloads" / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("comparison.csv", "predictions.csv", "coefficients.csv", "report.md", "results.json", "feature_comparison.png", "prediction_diagnostics.png", "learning_curve.png"):
            shutil.copy2(root / "outputs" / name / filename, destination / filename)
    (target / ".nojekyll").touch()
