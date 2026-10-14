# Boston Housing 分析結果

## 問題與資料
預測目標：`MEDV`；單位：千美元。
資料：506 筆、13 個原始特徵；編碼後 13 個特徵。
來源：https://www.openml.org/data/v1/download/52643/boston.arff
SHA-256：`2826ddb8cbce925e98114491f4774544b17d70ff0b7b1eede2af0a2ffd85a4f0`

## 方法
固定 random_state=0，80% 訓練、20% 最終測試。
比較 Lasso、Forward SFS、RFE、ElasticNet、Exhaustive Search 五種特徵選擇方法，另保留全特徵 OLS 基準。
訓練集 5-fold CV 選擇方法、特徵數與參數；Forward Selection 的每一步使用內層 CV。
缺值填補、類別編碼與標準化只學習當次訓練 fold。RFE 逐次移除標準化係數較小的特徵。
Lasso / ElasticNet 依標準化係數絕對值排序；固定 k 可能納入零係數尾端。五種方法的最終預測皆由選定子集合重新訓練 LinearRegression。
Exhaustive Search 實際枚舉 8191 個非空子集合，用 training CV 選擇各 k 的最佳子集合；超過 13 個編碼後特徵時停用此方法。
CV 分數是用於選擇的分數，尤其大量子集合搜尋可能造成選擇偏差；獨立 test set 保留到最後評估。

## 實際結果
CV 選定：**RFE (Backward)，11 個特徵**。
特徵：LSTAT, RM, PTRATIO, DIS, NOX, TAX, RAD, ZN, CRIM, B, CHAS。
CV RMSE：4.606036（fold 標準差 0.777782）。
Test MAE：3.842935；MSE：33.424197；RMSE：5.781366；R²：0.589527。
Test Adjusted R²：0.5393577878106651；Training OLS F-statistic：121.33898084982243；F p-value：8.152752930922365e-119。
全特徵基準 Test RMSE：5.783509。
本次選定模型的測試誤差低於全特徵基準；這是這次保留測試集上的觀察，不保證其他切分也相同。

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
