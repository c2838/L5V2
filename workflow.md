# 可重用的多元線性回歸 workflow

這份作業先實作 50 Startups，再把相同流程套用到 Boston Housing。第二題沒有另寫一套模型邏輯。

1. 定義資料來源、數值目標與單位；`datasets.py` 的 preset 是兩題唯一不同的設定入口。
2. 載入並記錄原始資料的 SHA-256；檢查筆數、目標型態、缺值、文字類別與無限值。
3. 固定 seed=0，保留 20% 為最終測試資料。
4. 在訓練資料內做 5-fold CV。每個 fold 重新學習缺值填補與 One-hot encoding。
5. 比較全特徵 OLS 基準及五種特徵選擇：Lasso、Forward SFS、RFE、ElasticNet、Exhaustive Search。Lasso / ElasticNet / RFE 的標準化只在當次訓練 fold 學習；Forward 的候選評估使用內層 CV，前處理也在內層 fit。
6. 依 training CV RMSE 選擇方法、特徵數與 alpha，再於完整 training set 重新選擇特徵與訓練。
7. 評估保留 test set，匯出指標、OLS 係數統計、殘差診斷、表格、圖表、模型、分析報告與靜態網頁。
8. 核對 test/train 無交集、序列化前後預測一致、網頁係數試算與 Python 一致。

## 第二題的直接套用

```sh
.venv/bin/python run.py --dataset startup
.venv/bin/python run.py --dataset boston
```

兩個入口都使用 `analysis.analyze` 與 `export.export_result`。Boston 的目標是 MEDV，單位為千美元；Startup 的目標是 Profit。

## 換成其他 Dataset

```sh
.venv/bin/python run.py --dataset custom --csv '<本機CSV路徑或公開HTTPS網址>' --target '<數值目標欄位>'
```

自訂 CSV 支援 25 筆以上有效資料、1～20 個原始特徵、編碼後最多 30 個特徵；每個文字欄位最多 12 個類別。應先排除公司編號、名稱等不適合預測的識別欄位。

自訂題目加入網頁下拉選單，Python 重新計算後才更新網頁。這不是在靜態網頁中直接訓練 sklearn。

Exhaustive Search 僅用 training CV 搜尋，最多 13 個編碼後特徵。OLS 的 F/t/p/CI/VIF 來自 training fit，Test Adjusted R² 則來自 test；統計推論以已選特徵為條件，未修正選擇影響。
