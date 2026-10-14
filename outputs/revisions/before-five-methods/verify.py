"""有意義的回歸驗證：真實產出、瀏覽器係數、資料洩漏與失敗路徑。"""
from __future__ import annotations

import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from scipy import stats
from itertools import combinations

from analysis import FixedSubsetRegressor, SELECTION_METHODS, analyze, best_sort_key, metrics
from datasets import download, load_dataset

ROOT = Path(__file__).resolve().parent


class RealArtifactTests(unittest.TestCase):
    def test_two_real_datasets_and_every_browser_prediction(self):
        """獨立重建每一列模型，再驗證匯出值及 JS 使用的原始單位公式。"""
        for name, count in [("startup", 50), ("boston", 506)]:
            with self.subTest(dataset=name):
                folder = ROOT / "outputs" / name
                result = json.loads((folder / "results.json").read_text())
                self.assertEqual(result["metadata"]["rows"], count)
                self.assertEqual(result["encoded_feature_count"], 5 if name == "startup" else 13)
                self.assertEqual(tuple(result["available_methods"]), SELECTION_METHODS)
                self.assertEqual(len(result["rows"]), 1+5*result["encoded_feature_count"])
                for method in SELECTION_METHODS:
                    self.assertEqual([r["k"] for r in result["rows"] if r["method"]==method], list(range(1,result["encoded_feature_count"]+1)))
                frame = pd.read_csv(ROOT / "data" / name / "cleaned.csv")
                X, y = frame.drop(columns=result["metadata"]["target"]), frame[result["metadata"]["target"]]
                xt, xv, yt, yv = train_test_split(X, y, test_size=result["test_size"], random_state=result["seed"])
                self.assertFalse(set(xt.index) & set(xv.index))
                self.assertEqual(result["best_key"], min(result["rows"], key=best_sort_key)["key"])
                for row in result["rows"]:
                    model = FixedSubsetRegressor(tuple(row["features"])).fit(xt, yt)
                    expected = model.predict(xv)
                    exported = np.array([p["predicted"] for p in row["predictions"]])
                    np.testing.assert_allclose(expected, exported, rtol=1e-10, atol=1e-7)
                    formula = np.full(len(xv), row["model"]["intercept"])
                    for term in row["model"]["terms"]:
                        values = xv[term["source"]].fillna(term["fill"])
                        if term["kind"] == "category":
                            values = values.astype(str).eq(term["value"]).astype(float)
                        formula += term["coefficient"] * values.to_numpy(dtype=float)
                    np.testing.assert_allclose(formula, expected, rtol=1e-10, atol=1e-7)
                    recomputed = metrics(yv, exported)
                    for metric, value in recomputed.items():
                        np.testing.assert_allclose(value, row["test"][metric], rtol=1e-10, atol=1e-7)
                with gzip.open(folder / "model.pkl.gz", "rb") as file:
                    restored = pickle.load(file)
                best = next(r for r in result["rows"] if r["key"] == result["best_key"])
                np.testing.assert_allclose(restored.predict(xv), [p["predicted"] for p in best["predictions"]], rtol=1e-10, atol=1e-7)

    def test_site_contains_real_results_and_downloads(self):
        javascript = (ROOT / "docs" / "data.js").read_text()
        result = json.loads(javascript.removeprefix("window.AIOT_RESULTS = ").strip().removesuffix(";"))
        self.assertTrue({"startup", "boston"}.issubset({d["metadata"]["name"] for d in result}))
        for data in result:
            for filename in ["comparison.csv", "coefficients.csv", "report.md", "results.json", "learning_curve.png"]:
                self.assertTrue((ROOT / "docs" / "downloads" / data["metadata"]["name"] / filename).is_file())

    def test_ols_statistics_against_independent_linear_regression(self):
        result=json.loads((ROOT/'outputs/startup/results.json').read_text())
        frame=pd.read_csv(ROOT/'data/startup/cleaned.csv')
        X=frame.drop(columns='Profit'); y=frame['Profit']
        xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=0)
        row=next(r for r in result['rows'] if r['key']=='lasso-1')
        feature=row['features'][0]
        reference=stats.linregress(xt[feature],yt)
        slope=row['statistics']['coefficients'][1]
        np.testing.assert_allclose(slope['std_error'],reference.stderr,rtol=1e-10)
        np.testing.assert_allclose(slope['p_value'],reference.pvalue,rtol=1e-9)
        np.testing.assert_allclose(row['statistics']['coefficients'][0]['std_error'],reference.intercept_stderr,rtol=1e-10)
        np.testing.assert_allclose(row['statistics']['f_stat'],(reference.slope/reference.stderr)**2,rtol=1e-10)
        self.assertGreater(row['statistics']['f_df_residual'],0)

    def test_exhaustive_search_is_actual_minimum_for_each_k(self):
        result=json.loads((ROOT/'outputs/startup/results.json').read_text())
        frame=pd.read_csv(ROOT/'data/startup/cleaned.csv')
        X=frame.drop(columns='Profit'); y=frame['Profit']
        xt,_,yt,_=train_test_split(X,y,test_size=.2,random_state=0)
        cv=KFold(5,shuffle=True,random_state=0)
        for k in range(1,6):
            scores=[]
            for features in combinations(result['encoded_features'],k):
                score=-cross_val_score(FixedSubsetRegressor(features),xt,yt,cv=cv,scoring='neg_root_mean_squared_error').mean()
                scores.append(score)
            row=next(r for r in result['rows'] if r['key']==f'exhaustive-{k}')
            np.testing.assert_allclose(row['cv_rmse'],min(scores),rtol=1e-10)
        self.assertEqual(result['exhaustive_search']['subsets_evaluated'],31)


class EdgeCaseTests(unittest.TestCase):
    def test_synthetic_custom_dataset_with_missing_values(self):
        # 此項是合成資料 smoke test；兩個正式題目仍由 RealArtifactTests 驗證。
        rng = np.random.default_rng(42)
        amount = rng.normal(50, 10, 60)
        category = np.array(["A", "B"] * 30, dtype=object)
        target = 3 * amount + (category == "B") * 10 + rng.normal(0, 2, 60)
        frame = pd.DataFrame({"amount": amount, "group": category, "answer": target})
        frame.loc[[1, 6, 20], "amount"] = np.nan
        frame.loc[[3, 17], "group"] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "synthetic.csv"
            frame.to_csv(csv, index=False)
            cleaned, metadata = load_dataset(root, "custom", str(csv), "answer")
            result, model, verification = analyze(cleaned, metadata, progress=lambda message: None)
            self.assertEqual(result["encoded_feature_count"], 2)
            self.assertEqual(metadata["missing_values"]["amount"], 3)
            self.assertEqual(metadata["missing_values"]["group"], 2)
            self.assertTrue(np.isfinite(model.predict(verification["X_test"])).all())
            self.assertEqual(result["metadata"]["target"], "answer")

    def test_imputer_uses_training_data_and_handles_unknown_category(self):
        X = pd.DataFrame({"amount": [1., 2., 3., np.nan], "state": ["A", "B", "A", "B"]})
        model = FixedSubsetRegressor().fit(X, np.array([3., 5., 7., 5.]))
        future = pd.DataFrame({"amount": [1e9, np.nan], "state": ["C", "A"]})
        prediction = model.predict(future)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(model.preprocessor_.named_transformers_["numeric"].statistics_[0], 2.)

    def test_invalid_dataset_rejected_with_clear_messages(self):
        scenarios = [
            (pd.DataFrame({"x": range(30), "y": range(30)}), "missing", "找不到目標欄位"),
            (pd.DataFrame({"x": range(30), "y": ["bad"]*30}), "y", "回歸目標必須"),
            (pd.DataFrame({"x": range(5), "y": range(5)}), "y", "至少需要 25 筆"),
            (pd.DataFrame({"x": [np.nan]*30, "y": range(30)}), "y", "完全沒有資料"),
            (pd.DataFrame({"x": range(30), "y": [1.]*30}), "y", "目標值全部相同"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, (frame, target, message) in enumerate(scenarios):
                with self.subTest(case=i):
                    csv = root / f"invalid-{i}.csv"
                    frame.to_csv(csv, index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        load_dataset(root, "custom", str(csv), target)

    def test_simulated_download_failure_and_invalid_url(self):
        # 此項明確為失敗注入；不把模擬測試當作真實網路下載。
        with patch("datasets.urlopen", side_effect=OSError("simulated network failure")):
            with self.assertRaisesRegex(ValueError, "資料下載失敗"):
                download("https://example.com/data.csv")
        with self.assertRaisesRegex(ValueError, "公開 HTTPS"):
            download("http://example.com/data.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
