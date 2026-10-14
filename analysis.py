"""以訓練集交叉驗證挑選特徵；保留的 test set 不參與選擇。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 0
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SELECTION_METHODS = ("lasso", "forward", "rfe", "elasticnet", "exhaustive")
METHODS = {"baseline": "Baseline OLS", "lasso": "Lasso (L1 Ranking)", "forward": "Forward SFS",
           "rfe": "RFE (Backward)", "elasticnet": "ElasticNet (L1 + L2)", "exhaustive": "Exhaustive Search (Best Subset)"}
L1_RATIOS = (0.2, 0.5, 0.8)


def best_sort_key(row):
    # 1e-9 的 CV 差異視為浮點誤差；同分時先選少特徵，再用固定方法順序。
    return (round(row["cv_rmse"], 9), row["k"], ("baseline", *SELECTION_METHODS).index(row["method"]))


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


def rank_elasticnet(matrix, y, names, alpha, l1_ratio):
    standardized = StandardScaler().fit_transform(matrix)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=30000, tol=1e-6).fit(standardized, y)
    return [names[i] for i in np.argsort(-np.abs(model.coef_), kind="stable")], model


def rank_rfe(matrix, y, names):
    standardized = StandardScaler().fit_transform(matrix)
    selector = RFE(LinearRegression(), n_features_to_select=1, step=1).fit(standardized, y)
    return [names[i] for i in np.argsort(selector.ranking_, kind="stable")]


def exhaustive_search(fold_matrices, names, progress):
    """枚舉全部非空子集合，依 training CV RMSE 為每個 k 挑出最佳子集合。"""
    selected, errors_by_k = {}, {}
    for k in range(1, len(names) + 1):
        best_score, best_indices, best_errors = np.inf, None, None
        for indices in combinations(range(len(names)), k):
            errors = []
            for mt, mv, yt, yv in fold_matrices:
                predicted = LinearRegression().fit(mt[:, indices], yt).predict(mv[:, indices])
                errors.append(float(np.sqrt(mean_squared_error(yv, predicted))))
            score = float(np.mean(errors))
            if score < best_score:
                best_score, best_indices, best_errors = score, indices, errors
        selected[k] = [names[i] for i in best_indices]
        errors_by_k[k] = best_errors
        progress(f"  Exhaustive Search：k={k}/{len(names)} 已完成")
    return selected, errors_by_k


def ols_statistics(estimator, X_train, y_train, test_r2, test_rows):
    """OLS 條件式統計：training fit 的 t/F/CI/VIF 與 test adjusted R²。"""
    matrix = matrix_for_names(estimator.preprocessor_, X_train, estimator.features_)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    rank = int(np.linalg.matrix_rank(design))
    dof = len(matrix) - rank
    residual = np.asarray(y_train) - estimator.predict(X_train)
    sse = float(residual @ residual)
    sst = float(np.sum((np.asarray(y_train) - np.mean(y_train))**2))
    full_rank = rank == design.shape[1]
    coefficients = np.r_[estimator.model_.intercept_, estimator.model_.coef_]
    table = []
    standard_errors = None
    if dof > 0 and full_rank:
        pinv = np.linalg.pinv(design)
        standard_errors = np.sqrt(np.maximum(np.diag((sse / dof) * (pinv @ pinv.T)), 0))
    for i, (name, coefficient) in enumerate(zip(["Intercept", *estimator.features_], coefficients)):
        se = float(standard_errors[i]) if standard_errors is not None else None
        t_value = float(coefficient / se) if se is not None and se > 0 else None
        p_value = float(2 * stats.t.sf(abs(t_value), dof)) if t_value is not None else None
        interval = [float(coefficient - stats.t.ppf(.975, dof) * se),
                    float(coefficient + stats.t.ppf(.975, dof) * se)] if se is not None else None
        vif, vif_infinite = None, False
        if i:
            index = i - 1
            target = matrix[:, index]
            others = np.delete(matrix, index, axis=1)
            if np.var(target) <= np.finfo(float).eps:
                vif_infinite = True
            elif not others.shape[1]:
                vif = 1.0
            else:
                r2 = LinearRegression().fit(others, target).score(others, target)
                if 1 - r2 <= 1e-12:
                    vif_infinite = True
                else:
                    vif = float(1 / (1 - r2))
        table.append({"predictor": name, "estimate": float(coefficient), "std_error": se,
                      "t_stat": t_value, "p_value": p_value, "ci95": interval,
                      "vif": vif, "vif_infinite": vif_infinite})
    f_stat = float(((sst-sse)/(rank-1)) / (sse/dof)) if rank > 1 and dof > 0 and sse > 0 else None
    f_p = float(stats.f.sf(max(f_stat, 0), rank-1, dof)) if f_stat is not None else None
    k = len(estimator.features_)
    adjusted = float(1-(1-test_r2)*(test_rows-1)/(test_rows-k-1)) if test_rows > k+1 and full_rank else None
    return {"test_adjusted_r2": adjusted, "f_stat": f_stat, "f_p_value": f_p,
            "f_df_model": rank-1, "f_df_residual": dof, "full_rank": full_rank,
            "coefficients": table,
            "scope": "F, t, CI and VIF use training OLS. Adjusted R² uses the held-out test set. Inference is conditional on selected predictors and is not adjusted for feature selection."}


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
    if not 1 <= p <= 30:
        raise ValueError("前處理後需要 1～30 個可用特徵。")
    methods = list(SELECTION_METHODS if p <= 13 else SELECTION_METHODS[:-1])
    fold_errors = {("baseline", p, None): []}
    for k in range(1, p+1):
        for method in ("forward", "rfe"):
            fold_errors[(method, k, None)] = []
        for alpha in ALPHAS:
            fold_errors[("lasso", k, alpha)] = []
            for ratio in L1_RATIOS:
                fold_errors[("elasticnet", k, (alpha, ratio))] = []
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_matrices = []
    for fold, (train_indices, valid_indices) in enumerate(cv.split(X_train), 1):
        progress(f"  CV {fold}/5：Lasso / Forward / RFE / ElasticNet")
        xt, xv = X_train.iloc[train_indices], X_train.iloc[valid_indices]
        yt, yv = y_train.iloc[train_indices], y_train.iloc[valid_indices]
        prep = make_preprocessor(xt).fit(xt)
        mt, mv = matrix_for_names(prep, xt, names), matrix_for_names(prep, xv, names)
        fold_matrices.append((mt, mv, yt, yv))
        baseline = LinearRegression().fit(mt, yt)
        fold_errors[("baseline", p, None)].append(metrics(yv, baseline.predict(mv))["rmse"])
        def evaluate_order(method, order, parameter):
            for k in range(1, p+1):
                indices = [names.index(n) for n in order[:k]]
                predicted = LinearRegression().fit(mt[:, indices], yt).predict(mv[:, indices])
                fold_errors[(method, k, parameter)].append(float(np.sqrt(mean_squared_error(yv, predicted))))
        for alpha in ALPHAS:
            order, _ = rank_lasso(mt, yt, names, alpha)
            evaluate_order("lasso", order, alpha)
            for ratio in L1_RATIOS:
                order, _ = rank_elasticnet(mt, yt, names, alpha, ratio)
                evaluate_order("elasticnet", order, (alpha, ratio))
        evaluate_order("rfe", rank_rfe(mt, yt, names), None)
        order = forward_order(xt, yt, names)
        for k in range(1, p+1):
            predicted = FixedSubsetRegressor(tuple(order[:k])).fit(xt, yt).predict(xv)
            fold_errors[("forward", k, None)].append(metrics(yv, predicted)["rmse"])
    exhaustive_subsets = {}
    if "exhaustive" in methods:
        progress(f"  Exhaustive Search：枚舉 {2**p-1} 個非空特徵子集合")
        exhaustive_subsets, exhaustive_errors = exhaustive_search(fold_matrices, names, progress)
        for k, errors in exhaustive_errors.items():
            fold_errors[("exhaustive", k, None)] = errors
    matrix = matrix_for_names(full_preprocessor, X_train, names)
    progress("  重新訓練各候選模型，計算保留測試集與 OLS 統計")
    full_forward = forward_order(X_train, y_train, names, folds=5)
    full_rfe = rank_rfe(matrix, y_train, names)
    rows, models, ranking_details = [], {}, {}
    for method in ["baseline", *methods]:
        for k in ([p] if method == "baseline" else range(1, p+1)):
            parameter = None
            alpha = ratio = None
            if method == "lasso":
                parameter = min(ALPHAS, key=lambda a: np.mean(fold_errors[(method,k,a)]))
                alpha = parameter
                order, lasso = rank_lasso(matrix, y_train, names, alpha)
                selected = order[:k]
                ranking_details[str(alpha)] = {"nonzero_count": int(np.count_nonzero(lasso.coef_)),
                    "coefficients": {n: float(c) for n,c in zip(names,lasso.coef_)}}
            elif method == "elasticnet":
                parameter = min(((a,r) for a in ALPHAS for r in L1_RATIOS),
                                key=lambda ar: np.mean(fold_errors[(method,k,ar)]))
                alpha, ratio = parameter
                order, _ = rank_elasticnet(matrix, y_train, names, alpha, ratio)
                selected = order[:k]
            elif method == "exhaustive":
                selected = exhaustive_subsets[k]
            else:
                selected = (names if method=="baseline" else full_forward if method=="forward" else full_rfe)[:k]
            estimator = FixedSubsetRegressor(tuple(selected)).fit(X_train, y_train)
            train_metrics = metrics(y_train, estimator.predict(X_train))
            test_metrics = metrics(y_test, estimator.predict(X_test))
            errors = fold_errors[(method,k,parameter)]
            key = f"{method}-{k}"
            row = {"key":key, "method":method, "method_label":METHODS[method], "k":k,
                   "features":selected, "alpha":alpha, "l1_ratio":ratio,
                   "cv_rmse":float(np.mean(errors)), "cv_rmse_std":float(np.std(errors,ddof=1)),
                   "train":train_metrics, "test":test_metrics, "model":browser_model(estimator,X_train),
                   "statistics":ols_statistics(estimator,X_train,y_train,test_metrics["r2"],len(X_test))}
            predicted = estimator.predict(X_test)
            row["predictions"] = [{"id":str(i), "actual":float(a), "predicted":float(b),
                "residual":float(a-b), "absolute_error":float(abs(a-b)),
                "relative_error_percent":float(abs(a-b)/abs(a)*100) if a else None,
                "diagnostic":"Above RMSE" if abs(a-b)>test_metrics["rmse"] else "Within RMSE"}
                for i,a,b in zip(X_test.index,y_test,predicted)]
            rows.append(row)
            models[key]=estimator
    best = min(rows,key=best_sort_key)
    progress(f"  CV 選定：{best['method_label']}，{best['k']} 個特徵")
    sizes,training_scores,validation_scores = learning_curve(FixedSubsetRegressor(),X_train,y_train,cv=cv,
        train_sizes=np.linspace(.35,1.,5),scoring="neg_root_mean_squared_error",error_score="raise")
    curve={"description":"All-feature baseline / 全特徵基準：僅使用訓練集內的 5-fold CV。",
           "sizes":sizes.tolist(),"train_rmse":(-training_scores).mean(axis=1).tolist(),
           "validation_rmse":(-validation_scores).mean(axis=1).tolist()}
    result={"metadata":metadata,"seed":SEED,"test_size":.2,"train_rows":len(X_train),"test_rows":len(X_test),
        "cv_folds":5,"encoded_feature_count":p,"encoded_features":names,"available_methods":methods,
        "input_schema":input_schema(X_train),"rows":rows,"best_key":best["key"],
        "cv_best_keys":[r["key"] for r in rows if round(r["cv_rmse"],9)==round(best["cv_rmse"],9) and r["k"]==best["k"]],
        "learning_curve":curve,"lasso_details":ranking_details,
        "exhaustive_search":{"enabled":p<=13,"subsets_evaluated":2**p-1 if p<=13 else 0,
            "cv_fit_count":5*(2**p-1) if p<=13 else 0,"selection_basis":"training 5-fold CV RMSE",
            "limit_reason":None if p<=13 else "More than 13 encoded features: exhaustive enumeration disabled."},
        "preview":frame.head(5).replace({np.nan:None}).to_dict(orient="records")}
    verification={"test_indices":X_test.index.tolist(),"train_indices":X_train.index.tolist(),"X_test":X_test,"y_test":y_test}
    return result,models[best["key"]],verification
