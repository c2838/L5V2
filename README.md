# Multiple Linear Regression Explorer｜AIoT DIC2

用 Python / scikit-learn 完成 50 Startups 與 Boston Housing，產生可放 GitHub Pages 的靜態互動網頁。主程式為 `.py`，不依賴 Notebook。

## 執行與查看

專案路徑：`/Users/c2838/NCHU/115-1 中興作業/AIOT/L5V2`。使用 Python 3.12，已建立 `.venv`；實際套件版本固定於 `requirements.txt`。

```sh
cd '/Users/c2838/NCHU/115-1 中興作業/AIOT/L5V2'
.venv/bin/python run.py --dataset all
```

第一次需要網路下載，之後使用 `data/` 的副本；`--refresh` 重新下載。兩題共用同一個 `analyze` workflow，第二題只更換資料與目標設定。

用瀏覽器開啟 `docs/index.html`，或啟動本機 HTTP 預覽：

```sh
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

開啟 `http://127.0.0.1:8765`。只重建網站使用 `.venv/bin/python run.py --site-only`。

新電腦：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 與老師一致的結果種類

- **五種特徵選擇方法**：Lasso (L1 Ranking)、Forward SFS、RFE (Backward)、ElasticNet (L1 + L2)、Exhaustive Search (Best Subset)。Baseline OLS 另列，不算五種方法之一。
- **Compare All 5 Methods**：同圖比較五種方法跨 k 的 Test RMSE、Test R²；同一 k 的比較表包含特徵名稱、CV RMSE、Test RMSE、Test R²。
- **五個主要指標**：Test RMSE、Test R²、Test Adjusted R²、Test MAE、Training OLS F-statistic 與 p-value。Test MSE 另保留於結果摘要、CSV 與 JSON。
- **Fitted OLS Equation**：原始單位的截距與係數。
- **Model Coefficients & Statistical Tests**：Estimate、Std. Error、t-stat、p-value、95% CI、VIF。
- **Live Predictor**：輸入數值 / 類別，以 Python 匯出的完整精度係數計算；不重新訓練。
- **Residuals & Error Diagnostics**：依絕對殘差排序，顯示 Sample ID、Actual、Predicted、Residual、Abs. Error、Error %、描述性 Status。
- 另保留 Actual vs. Predicted、Residuals vs. Predicted 與 baseline Learning Curve。
- 圖表標題、座標軸與圖例以英文為主、中文輔助，英文字級較大。
- 資料集選單保留於頁首，預設 50 Startups。LMS 要求兩題都做，並列有題目切換加分；老師示範頁面只展示 50 Startups。

## 實際方法與資料獨立性

固定 `random_state=0`，80% training、20% final test。training 5-fold CV 選方法、特徵數與參數；test 不參與選擇。相差小於約 1e-9 的 CV 分數視為浮點同分，以少特徵及固定方法順序決定顯示的代表模型。

前處理於每個 training fold 學習：數值缺值填補、類別填補、One-hot encoding（drop first）。Lasso、ElasticNet 與 RFE 的標準化也只使用當次訓練資料。

- Lasso / ElasticNet：按標準化係數絕對值排序；各 k 的 alpha / l1_ratio 由 training CV 選擇。固定 k 可能納入排序尾端的零係數特徵。
- Forward SFS：內層 CV 逐次選擇新增特徵。
- RFE：以 sklearn RFE 遞迴移除標準化 OLS 權重較小的特徵。
- Exhaustive Search：實際枚舉所有非空子集合，用 training CV 為每個 k 挑出最佳子集合。Startup 搜尋 31 個子集合；Boston 搜尋 8191 個子集合。超過 13 個編碼後特徵時停用此方法，避免指數運算量。
- 五種方法最後都以所選特徵重新訓練 `LinearRegression`（OLS）；表格、公式與統計均描述這個最終 OLS 模型。

大量參數與子集合搜尋後的 CV 是選擇分數，不宣稱為無偏泛化估計；最終獨立 test 結果仍保留。方法可能選到相同最終特徵，產生相同 test 分數與重疊曲線；不同 fold 的選擇過程仍可能有不同 CV 分數。沒有為了讓方法看起來不同而改動數值。

## 統計結果的範圍

Adjusted R² 使用 test R²、test 樣本數 n、所選特徵數 k：`1 - (1 - R²) × (n - 1) / (n - k - 1)`。若自由度不足或模型非滿秩，顯示不可計算。

F/t/p/95% CI/VIF 使用 **training OLS**。標準誤由 training 殘差變異與設計矩陣求得；需符合 OLS 的統計假設。這些推論以已選定變數為條件，未修正特徵選擇影響，因此作探索性說明，不能當作因果證據。

Residual = Actual - Predicted。Status 的 `Above RMSE` 代表絕對誤差超過目前 Test RMSE，僅是描述性標記，不是正式離群檢定。Sample ID 是原始資料的零起算列索引。相對誤差遇到 Actual=0 時顯示不可計算。

Learning Curve 是全特徵 OLS 基準的 training/validation RMSE，不混用最終 test。MAE/RMSE 與目標同單位，MSE 是平方單位；R² 不是分類準確率。

## 資料與自訂 Dataset

Startup 使用 LMS 提供的 Avik Jain `50_Startups.csv`，目標 Profit。Boston 使用 OpenML data_id=531、version=1，目標 MEDV、單位千美元。

Boston ARFF 的 CHAS/RAD 宣告為 nominal，讀入會成為 bytes；載入時解碼並還原數值，保留原始 13 個特徵。資料原文、來源與 SHA-256 皆存於 `data/`。

```sh
.venv/bin/python run.py --dataset custom --csv '<CSV路徑或公開HTTPS網址>' --target '<數值目標欄位>'
```

自訂資料至少 25 筆有效資料、1～20 個原始特徵、最多 30 個編碼後特徵；文字欄位最多 12 個類別。CSV 重新計算後才更新網頁，不是公開靜態頁面即時訓練 sklearn。

兩題誤差單位不同，不能直接比較 RMSE。Boston 具有歷史與偏差限制，本作業只作教學分析。單次切分、小樣本與選擇程序限制結果穩定性；不以老師示範數字作為通過標準。

## 產出與驗證

- `outputs/startup/`、`outputs/boston/`：完整 JSON、比較表、預測 CSV、係數 CSV、圖表、最佳模型、分析報告與驗證紀錄。
- `docs/`：靜態成果與下載檔；不包含虛擬環境及 pickle 模型。
- `workflow.md`：第二題重用方式。
- `VALIDATION.md`：最新實際測試結果與限制。
- `outputs/revisions/before-five-methods/`：修改前的原始檔紀錄。

```sh
.venv/bin/python verify.py
```

驗證全部 92 組模型與原始單位網頁公式；以 scipy 單變數回歸獨立核對標準誤、p/F/CI，以相關係數核對雙變數 VIF；對 Startup 的全部子集合獨立驗證 Exhaustive Search。

## 架構

GitHub Pages 不能直接執行 sklearn，因此 Python 分析、瀏覽器呈現。`data.js` 避免雙擊 HTML 時 fetch JSON 的限制。類別的係數映射依 encoder 原生輸出建立，不靠切割欄位字串。

本機版尚未公開發布、推送 repository 或繳交 LMS。

參考：[老師成果網站](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)、[sklearn RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)、[sklearn ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)。
