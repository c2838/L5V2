# 回歸研究室｜AIoT DIC2

使用 Python / scikit-learn 完成 50 Startups 與 Boston Housing 的多元線性回歸分析，並產生可放在 GitHub Pages 的靜態互動網頁。主程式為 `.py`，不依賴 Notebook。

## 執行

專案路徑：`/Users/c2838/NCHU/115-1 中興作業/AIOT/L5V2`。

已建立專案內 `.venv`，使用 Python 3.12。套件版本固定於 `requirements.txt`。

```sh
cd '/Users/c2838/NCHU/115-1 中興作業/AIOT/L5V2'
.venv/bin/python run.py --dataset all
```

第一次下載資料需要網路，之後使用 `data/` 下的本機副本。`--refresh` 可更新預設資料。兩題分析皆包含內層交叉驗證，執行時間依電腦效能而異。

更換電腦時建立環境：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 查看成果

直接用瀏覽器開啟 `docs/index.html`，不需要 Python 伺服器即可切換兩題、方法及特徵數，查看圖表與試算預測。

也可在本機啟動靜態預覽：

```sh
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

然後開啟 `http://127.0.0.1:8765`。此操作只服務本機。

網站展示預先計算的結果；單筆預測以匯出的線性係數與原始欄位映射計算。網頁不重訓 sklearn。更換 CSV 後需重新執行 Python：

```sh
.venv/bin/python run.py --dataset custom --csv '<CSV路徑或公開HTTPS網址>' --target '<目標欄位>'
```

只重建網頁、不重新運算：

```sh
.venv/bin/python run.py --site-only
```

## 方法與作業要求

- 全特徵基準使用 `sklearn.linear_model.LinearRegression`。
- Lasso 使用標準化後的係數絕對值排序。每個 k 的 alpha 由 training CV 選擇，再以選定特徵重新訓練 LinearRegression。固定 k 的後段可能包含 Lasso 零係數特徵，實際非零數量另外記錄於 results.json。
- Forward Selection 在內層 CV 評估每次新增的候選特徵，於外層 training CV 比較不同 k。
- `State` 採 One-hot encoding 並捨棄一個參考類別，避免與截距形成線性相依。
- 先切分 80/20，再在 training set 做 5-fold CV。填補、編碼與 Lasso 標準化在各 fold 內 fit。最佳模型由 CV RMSE 選定，不用 test 選擇。
- 保留 test 的 MAE、MSE、RMSE、R²；training/validation learning curve 為全特徵基準模型。
- 以共用 workflow 套用第二題，參見 `workflow.md`。
- 特徵數指編碼後欄位；不是原始欄位數。Startup 預設有 5 個編碼後特徵，Boston 13 個。
- 存下各題最佳模型與前處理器（gzip + pickle），供同一專案環境載入驗證。

## 資料與解讀

Startup 使用 LMS 提供的 Avik Jain 50_Startups.csv；Boston 使用 OpenML data_id=531、version=1，目標 MEDV，單位千美元。

MAE、RMSE 與目標同單位，MSE 是平方單位；R² 不是分類準確率。係數不代表因果。兩題單位不同，不直接比較 RMSE。Boston 為歷史教學資料，具有時代與偏差限制，不作實際房價決策。

老師示範中的數字不作為通過標準。使用相同原始資料，也可能因切分、特徵選擇與驗證流程而得到不同結果。

## 產出

- `outputs/startup/`、`outputs/boston/`：JSON、比較表、測試預測、三張 PNG 圖、最佳模型、分析報告與驗證紀錄。
- `docs/`：可直接開啟的靜態網站及下載檔；不包含 pickle 模型或虛擬環境。
- `data/`：原始資料、清理後 CSV、來源與 SHA-256 紀錄。
- `VALIDATION.md`：實際測試方法、結果與限制。

## 架構選擇

GitHub Pages 不能直接執行 sklearn，所以 Python 負責分析，瀏覽器只呈現 JSON 結果與係數試算。使用 `data.js` 載入資料，讓直接雙擊 `index.html` 也能開啟，避免 file:// 的 fetch 限制。

數值與類別的前處理映射隨模型一起匯出；類別名稱不靠字串切割，避免欄位名稱包含底線時映射錯誤。不同 CV fold 的欄位對齊由名稱處理。

本機成果完成後，若要公開發布，需另外確認 repository 與 GitHub Pages 設定；本專案目前不包含自動推送或部署步驟。
