# 五方法與完整統計結果驗證報告

日期：2026-10-14（Asia/Taipei）。本次版本補齊老師範例的五種特徵選擇與完整結果種類；上一版文件留於 `outputs/revisions/before-five-methods/`。

## 實際資料與模型

公開資料副本實際以 Python / sklearn 執行，沒有 mock 指標。

| 題目 | 筆數 | Train / Test | 編碼後特徵 | 候選模型 | Training CV 選定 | Test RMSE | Test R² | Test Adjusted R² | Training F |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 50 Startups | 50 | 40 / 10 | 5 | 26 | Lasso, k=2 | 8198.797191 | 0.947439 | 0.932421 | 349.041043 |
| Boston Housing | 506 | 404 / 102 | 13 | 66 | RFE, k=11 | 5.781366 | 0.589527 | 0.539358 | 121.338981 |

兩題各有五種選擇方法 × 全部 k，加一個 Baseline OLS，共 92 組模型。Baseline 不算五種方法之一。

Startup 使用 R&D Spend、Marketing Spend。兩特徵 training OLS 的 Intercept=46346.3388，R&D estimate≈0.7886，Marketing estimate≈0.0326；其標準誤、t/p、95% CI、VIF 與老師展示的數值相符。

新加入的方法使 Boston 的 training CV 選擇改為 RFE、11 個特徵；不沿用上一版的 13 特徵最佳標籤。

## 方法與統計範圍

五方法：Lasso (L1 Ranking)、Forward SFS、RFE (Backward)、ElasticNet (L1 + L2)、Exhaustive Search (Best Subset)。全部以所選欄位重新訓練最終 OLS，故相同子集合可能有相同 test 指標。

- 外層 training 5-fold CV 使用相同切分；填補、編碼與標準化於當次 training fold 學習。
- Forward 的候選評估另做內層 CV。
- Exhaustive 實際枚舉 Startup 的 31 個、Boston 的 8191 個非空子集合；分別共 155 / 40955 次 CV OLS fits。
- Training CV 分數用於方法、k、參數 / 子集合選擇，不宣稱搜尋後無偏；final test 不參與選擇。
- Test Adjusted R² 由 test R²、test n 與 k 求得。
- F/t/p/CI/VIF 由 training OLS 求得，使用 classical OLS 假設；推論以已選變數為條件，未做 post-selection correction。未實施正式殘差常態 / 等變異假設檢定，不將統計數字解讀成因果結論。
- 殘差=Actual−Predicted。Above RMSE 僅表示 |Residual| 大於當前 test RMSE，不是正式離群檢定。

## 自動測試：8 項通過

執行 `.venv/bin/python verify.py`，實際取得 `Ran 8 tests … OK`。

| 驗證方法 / 範圍 | 判斷基準 | 實際結果 |
|---|---|---|
| 正式兩題及五方法完整性 | 50 / 506 筆、5 / 13 特徵；每種方法涵蓋全部 k，26 / 66 候選 | 符合 |
| 全部 92 個候選模型獨立重建 | sklearn 預測與 JSON，rtol=1e-10 / atol=1e-7 | 92 組符合 |
| 原始單位網頁係數試算 | 包含 State 類別，與各模型 sklearn 預測一致 | 92 組符合 |
| 四個 test 指標重算 | 由匯出預測重新計算 MAE、MSE、RMSE、R² | 符合 |
| 最佳模型、索引、序列化 | 依 CV 同分規則選代表模型；train/test 不重疊；gzip/pickle 載入一致 | 符合 |
| 單特徵 OLS 統計獨立核對 | 用 scipy.stats.linregress 比對 slope/intercept SE、p、F=t²、95% CI | rtol 約 1e-9～1e-10 內符合 |
| 雙特徵 VIF 核對 | 由 training Pearson correlation 求 1/(1−r²) | rtol=1e-10 內符合 |
| Exhaustive Search 獨立核對 | Startup 全部 31 子集合用 cross_val_score 再算；每個 k 等於最小 CV RMSE | 符合 |
| 網頁成果 / 附件 | 兩題資料與 comparison / coefficients / report / JSON / 圖檔存在 | 符合 |
| 缺值與未知類別（小型測試） | training median 不因新極端值改變；未知類別仍可預測 | 符合，出現預期的 sklearn unknown-category warning |
| 自訂資料完整流程（合成資料） | 60 筆、有數值與類別缺值，同一分析流程可產生有限預測 | 符合；不當作第三份真實資料實驗 |
| 失敗路徑（模擬 / 測試 CSV） | 缺目標、文字目標、筆數不足、全空特徵、常數目標、非 HTTPS、下載例外應拒絕 | 符合；下載例外為明確模擬，不代表實際服務測試 |

## 重現性

新版兩題完整重跑一次，包含五方法 CV、OLS 統計、係數、預測的兩份 results.json 在重跑前後逐位相同（SHA-256 相同）。

## 瀏覽器實測

本機 HTTP 首頁回應 200，Chrome 實際載入新版。最終 console error 為空，JavaScript 語法檢查通過。

- 顯示五種方法與 Compare All 5 Methods；Baseline 另列。
- 逐一切換五種方法，等待狀態更新再核對；全部可更新目前方法與指標。
- Compare All 在 Startup 與 Boston 皆產生五條 RMSE 曲線與五列同 k 比較結果。
- Startup k=3 的比較列實際顯示五方法的特徵 / CV / test 結果；相同最後子集合產生相同 test RMSE，而 CV 值可不同。
- Dataset 切到 Boston，預設變為 RFE / k=11，Test R² 顯示 0.5895、Adjusted R² 0.5394、F 121.34，係數表含 Intercept＋11 欄，殘差表為 102 個 test samples 的前 10 筆。
- Startup k=2，指標顯示 RMSE 8198.80、R² 0.9474、Adjusted R² 0.9324、MAE 6886.59、F 349.04；係數表含三列，殘差表含十列。
- 英文圖表標題 17px、中文 11px；英文 SVG 軸標題 13px、中文 9px。手機版標題 / 圖例也保留英文較大。
- 390×844 手機版：document scrollWidth=390；所有指標數字未超出卡片寬度，沒有整頁橫向溢出。寬統計表使用自身橫向捲動。
- 新版比較 CSV 實際下載，與目前 26 列 Startup 匯出檔逐位相同，證據存於 `outputs/browser-checks/revision2/`。
- 新截圖：`outputs/preview/desktop-v2.jpg`、`comparison-v2.jpg`、`statistics-v2.jpg`、`mobile-v2.jpg`。

## 結果種類對照

已提供與老師相同的五主要指標、RMSE / R² 跨特徵曲線、OLS equation、係數 SE/t/p/CI/VIF 表、即時係數預測、殘差 / 絕對誤差 / 相對誤差 / Status 診斷表。另保留 MSE、Baseline、Learning Curve 與原有兩題資料集選單。

## 限制與未驗證

- 未公開發布、未推送 repository、未繳交 LMS。
- 未逐一點擊全部 92 個模型；Python 對全部模型核對，瀏覽器涵蓋五方法、同 k 比較、兩題與手機代表操作。
- 瀏覽器工具禁止 file://；本次透過 HTTP 驗證相同靜態檔案。使用者自行開啟的 HTML 頁面需重新整理載入新版。
- 自訂公開網址的可用性依來源而定；自訂完整流程實測合成 CSV，正式兩題使用真實公開資料。
- 超過 13 個編碼後特徵時停用 Exhaustive Search；不把其他方法偽裝成 exhaustive。
- 單次資料切分、小樣本與選擇程序限制結果的泛化解讀。曲線重疊和結果與範例不同皆可能是實際方法的結果，不強制改成範例數字。
