"""以訓練集交叉驗證挑選特徵；保留的 test set 不參與選擇。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 0
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
METHODS = {"baseline": "全特徵 LinearRegression", "lasso": "Lasso 排序＋LinearRegression", "forward": "Forward Selection＋LinearRegression"}


def make_preprocessor(X):
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    blocks = []
    if numeric:
        blocks.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        blocks.append(("category", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(blocks, verbose_feature_names_out=False)


def matrix_for_names(preprocessor, X, names):
    matrix = preprocessor.transform(X)
    available = list(preprocessor.get_feature_names_out())
    # 某個內層 CV fold 沒看過的類別欄位，對齊為 0。
    return np.column_stack([matrix[:, available.index(n)] if n in available else np.zeros(len(X)) for n in names])


class FixedSubsetRegressor(RegressorMixin, BaseEstimator):
    """每次 fit 都重新在當次訓練資料學習 imputer/encoder。"""
    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y):
        self.preprocessor_ = make_preprocessor(X)
        self.preprocessor_.fit(X)
        self.features_ = list(self.features) if self.features is not None else list(self.preprocessor_.get_feature_names_out())
        self.model_ = LinearRegression().fit(matrix_for_names(self.preprocessor_, X, self.features_), y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.model_.predict(matrix_for_names(self.preprocessor_, X, self.features_))


def metrics(y, prediction):
    mse = float(mean_squared_error(y, prediction))
    return {"mae": float(mean_absolute_error(y, prediction)), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y, prediction))}


def rank_lasso(matrix, y, names, alpha):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    model = Lasso(alpha=alpha, max_iter=30000, tol=1e-6).fit(standardized, y)
    order = np.argsort(-np.abs(model.coef_), kind="stable")
    # 固定 k 的比較包含零係數尾端；另匯出真正的非零係數數量。
    return [names[i] for i in order], model


def forward_order(X, y, candidates, folds=4):
    """每個候選的前處理皆在 inner CV 的 training fold 內 fit。"""
    chosen, remaining = [], list(candidates)
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    while remaining:
        scores = []
        for feature in remaining:
            errors = -cross_val_score(FixedSubsetRegressor(tuple(chosen + [feature])), X, y,
                                      cv=cv, scoring="neg_root_mean_squared_error", error_score="raise")
            scores.append(float(errors.mean()))
        next_feature = remaining[int(np.argmin(scores))]
        chosen.append(next_feature)
        remaining.remove(next_feature)
    return chosen


def input_schema(X):
    schema = []
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            schema.append({"name": col, "type": "number", "default": float(series.median()),
                           "min": float(series.min()), "max": float(series.max())})
        else:
            values = sorted(map(str, series.dropna().unique()))
            schema.append({"name": col, "type": "category", "default": str(series.mode().iloc[0]), "values": values})
    return schema


def browser_model(estimator, X_train):
    """匯出原始單位的 LR 係數，以及完整的缺值/類別轉換資訊。"""
    terms = []
    prep = estimator.preprocessor_
    mappings = {}
    for _, transformer, columns in prep.transformers_:
        if transformer == "drop":
            continue
        if isinstance(transformer, Pipeline):
            encoder = transformer.named_steps["encode"]
            filler = transformer.named_steps["fill"]
            for i, col in enumerate(columns):
                names = encoder.get_feature_names_out([*columns])
                # 利用 encoder 的原生輸出順序映射，不靠字串切割欄位名稱。
                offset = sum(len(c) - 1 for c in encoder.categories_[:i])
                for j, category in enumerate(encoder.categories_[i][1:]):
                    mappings[names[offset+j]] = {"source": col, "kind": "category", "value": str(category), "fill": str(filler.statistics_[i])}
        else:
            for i, col in enumerate(columns):
                mappings[col] = {"source": col, "kind": "number", "fill": float(transformer.statistics_[i])}
    for name, coefficient in zip(estimator.features_, estimator.model_.coef_):
        terms.append({"feature": name, "coefficient": float(coefficient), **mappings[name]})
    return {"intercept": float(estimator.model_.intercept_), "terms": terms}


def analyze(frame, metadata, progress=print):
    X = frame.drop(columns=metadata["target"])
    y = frame[metadata["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    full_preprocessor = make_preprocessor(X_train).fit(X_train)
    names = list(full_preprocessor.get_feature_names_out())
    p = len(names)
    if p > 30:
        raise ValueError("編碼後超過 30 個特徵，請先減少類別或欄位。")
    if p < 1:
        raise ValueError("前處理後沒有可用特徵。")
    fold_errors = {("baseline", p, None): []}
    for k in range(1, p + 1):
        fold_errors[("forward", k, None)] = []
        for alpha in ALPHAS:
            fold_errors[("lasso", k, alpha)] = []
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_indices, valid_indices) in enumerate(cv.split(X_train), 1):
        progress(f"  CV {fold}/5：Lasso 排序與 Forward Selection")
        xt, xv = X_train.iloc[train_indices], X_train.iloc[valid_indices]
        yt, yv = y_train.iloc[train_indices], y_train.iloc[valid_indices]
        prep = make_preprocessor(xt).fit(xt)
        available = list(prep.get_feature_names_out())
        mt = matrix_for_names(prep, xt, names)
        mv = matrix_for_names(prep, xv, names)
        baseline = LinearRegression().fit(mt, yt)
        fold_errors[("baseline", p, None)].append(metrics(yv, baseline.predict(mv))["rmse"])
        for alpha in ALPHAS:
            order, _ = rank_lasso(mt, yt, names, alpha)
            for k in range(1, p + 1):
                indices = [names.index(n) for n in order[:k]]
                prediction = LinearRegression().fit(mt[:, indices], yt).predict(mv[:, indices])
                fold_errors[("lasso", k, alpha)].append(metrics(yv, prediction)["rmse"])
        order = forward_order(xt, yt, names)
        for k in range(1, p + 1):
            prediction = FixedSubsetRegressor(tuple(order[:k])).fit(xt, yt).predict(xv)
            fold_errors[("forward", k, None)].append(metrics(yv, prediction)["rmse"])
    matrix = matrix_for_names(full_preprocessor, X_train, names)
    progress("  在完整訓練集建立各特徵組合，評估保留測試集")
    full_forward_order = forward_order(X_train, y_train, names, folds=5)
    rows, models = [], {}
    ranking_details = {}
    for method in ("baseline", "lasso", "forward"):
        for k in ([p] if method == "baseline" else range(1, p + 1)):
            alpha = min(ALPHAS, key=lambda a: np.mean(fold_errors[(method, k, a)])) if method == "lasso" else None
            if method == "lasso":
                order, lasso = rank_lasso(matrix, y_train, names, alpha)
                ranking_details[str(alpha)] = {"nonzero_count": int(np.count_nonzero(lasso.coef_)),
                    "coefficients": {n: float(c) for n, c in zip(names, lasso.coef_)}}
            else:
                order = names if method == "baseline" else full_forward_order
            selected = order[:k]
            estimator = FixedSubsetRegressor(tuple(selected)).fit(X_train, y_train)
            train_metrics, test_metrics = metrics(y_train, estimator.predict(X_train)), metrics(y_test, estimator.predict(X_test))
            errors = fold_errors[(method, k, alpha)]
            key = f"{method}-{k}"
            row = {"key": key, "method": method, "method_label": METHODS[method], "k": k,
                   "features": selected, "alpha": alpha, "cv_rmse": float(np.mean(errors)),
                   "cv_rmse_std": float(np.std(errors, ddof=1)), "train": train_metrics, "test": test_metrics,
                   "model": browser_model(estimator, X_train)}
            prediction = estimator.predict(X_test)
            row["predictions"] = [{"id": str(i), "actual": float(a), "predicted": float(b), "residual": float(a-b)}
                                   for i, a, b in zip(X_test.index, y_test, prediction)]
            rows.append(row)
            models[key] = estimator
    best = min(rows, key=lambda r: (r["cv_rmse"], r["k"]))
    progress(f"  CV 選定：{best['method_label']}，{best['k']} 個特徵")
    sizes, training_scores, validation_scores = learning_curve(
        FixedSubsetRegressor(), X_train, y_train, cv=cv,
        train_sizes=np.linspace(0.35, 1.0, 5), scoring="neg_root_mean_squared_error", error_score="raise")
    curve = {"description": "全特徵基準模型：僅使用訓練集內的 5-fold CV，前處理在每個 fold 內學習。",
             "sizes": sizes.tolist(), "train_rmse": (-training_scores).mean(axis=1).tolist(),
             "validation_rmse": (-validation_scores).mean(axis=1).tolist()}
    # 匯出每個候選的 test 結果供比較，但最佳標籤只根據 training CV。
    result = {"metadata": metadata, "seed": SEED, "test_size": 0.2,
              "train_rows": len(X_train), "test_rows": len(X_test), "cv_folds": 5,
              "encoded_feature_count": p, "encoded_features": names,
              "input_schema": input_schema(X_train), "rows": rows, "best_key": best["key"],
              "learning_curve": curve, "lasso_details": ranking_details,
              "preview": frame.head(5).replace({np.nan: None}).to_dict(orient="records")}
    verification = {"test_indices": X_test.index.tolist(), "train_indices": X_train.index.tolist(),
                    "X_test": X_test, "y_test": y_test}
    return result, models[best["key"]], verification
