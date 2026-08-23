# checkpoint 目錄檔案說明

**適用**：`checkpoints/stock_63_21_DDPM_stock_ftM_sl63_ll21_pl21_dt0_TWO`（原始，x_start）
和 `..._TWO_v`（v-parameterization）。兩者結構完全相同，各約 **1.9 GB**。

視窗設定 `seq_len 63 → pred_len 21`，測試期 **2020-01-02 ~ 2024-12-31**。

---

## 一、根目錄

### 模型權重

| 檔案 | 大小 | 說明 |
|---|---|---|
| `checkpoint.pth` | 16 MB | **主權重**。100 epoch 中 val loss 最低的一版，所有推論都載入它 |
| `pretrain_model_checkpoint.pth` | 3.9 KB | 預訓練的 DLinear（`nn.Linear(21→21)`）。它的線性預測會當成條件餵給擴散模型，所以很小 |

### 訓練過程

| 檔案 | 大小 | 說明 |
|---|---|---|
| `loss_curve.png` | 74 KB | train / val loss 隨 epoch 的曲線 |
| `losses.csv` | 2.5 KB | 同樣的數字，每 epoch 一列，方便自己畫 |

### 測試診斷

| 檔案 | 大小 | 說明 |
|---|---|---|
| `MTS_errors.png` | 26 KB | 30 個通道各自的誤差長條圖，看哪一檔最難預測 |
| `MTS_errors_hist.png` | 11 KB | 誤差的分布直方圖 |

### 預測輸出（步長 1）

以 **1 天為步長**滑動，1238 個**互相重疊**的 21 天窗口。這組是 MSE/MAE 指標的來源。

| 檔案 | 大小 | 說明 |
|---|---|---|
| `test_predictions.csv` | 11 MB | 25,998 列（1238 窗口 × 21 天）× 30 變數，含 `date` / `window` / `step` |
| `test_ground_truth.csv` | 11 MB | 對應真實值，同格式，可直接相減 |
| `test_arrays.npz` | 3.0 MB | 上面兩者的陣列版（`preds` `trues` `cols` `dates`），供離線重畫圖 |

### 分析圖（步長 1）

| 檔案 | 大小 | 說明 |
|---|---|---|
| `volatility_check.png` | 189 KB | 波動對比。10 檔同一張：水準、追蹤散點、時間軸 |
| `volatility_check_by_ticker.png` | 300 KB | 同上，10 檔分開（2×5），各自的波動走勢 |
| `distribution_check.png` | 132 KB | 分布對比：分布本體、尾部（對數軸）、Q-Q 圖、每檔峰態 |
| `distribution_check_by_ticker.png` | 170 KB | 同上，10 檔分開 |

> 縱軸用**該檔在測試期的實際波動**標準化，所以真實序列 sd 恰為 1.00，模型的 sd 直接讀成「達到市場波動的幾成」。

---

## 二、`price_plots/` 與 `knob_plots/`（各 20 檔）

步長 21、59 個不重疊區塊，連續複利畫法。每檔股票兩張圖。

| 檔案 | 說明 |
|---|---|
| `<TIC>_logret.png` | 對數報酬的預測 vs 真實 |
| `<TIC>_price.png` | 還原成價格的走勢，含 GBM 隨機漫步基準線 |

兩個目錄的差別只有一個：**`knob_plots/` 多一條橘線**，代表 `knob = -3` 的預測，用來跟不轉旋鈕的紅線對照。

---

## 三、`path_forecast*/` — 接成一條連續路徑（三個情境）

每個模型有三個情境目錄，各約 **600 MB**：

| 目錄 | 旋鈕 | guidance | 說明 |
|---|---|---|---|
| `path_forecast/` | 無 | — | 原本的兩條件模型 |
| `path_forecast_knob-3/` | −3 | 1.0 | 轉旋鈕，但 CFG 未開啟 |
| `path_forecast_knob-3_w5/` | −3 | 5.0 | 轉旋鈕並放大訊號 5 倍 |

### 做法

步長 21：預測 21 天 → 右移 21 天 → 再預測，走完測試期。60 個區塊首尾相接成 **1258 天**（2020-01-02 ~ 2024-12-31）。最後一塊只貢獻 19 天，補齊 1258 不是 21 倍數的尾巴。

價格是**一路複利、中途不重新描點**：

```
P_hat[t] = P_0 × exp( cumsum(r_close) )     P_0 = 測試期第一天前的真實收盤價
```

### 目錄內容

| 檔案 / 目錄 | 大小 | 說明 |
|---|---|---|
| `paths_prices/path_0001.csv` … `path_0500.csv` | 555 KB × 500 | **500 條生成路徑的股價** |
| `paths_simple_returns/path_0001.csv` … `path_0500.csv` | × 500 | **500 條生成路徑的簡單報酬** |
| `true_prices.csv` | 554 KB | 真實股價。**只存在無旋鈕目錄**（真實值不隨情境改變） |
| `true_simple_returns.csv` | 669 KB | 真實簡單報酬，同上 |
| `stitched_path.png` | 337 KB | 10 檔各一格，真實價格 vs 連續預測路徑（實線＝path 1，陰影＝500 條的 10-90%） |
| `path_volatility_check.png` | 164 KB | 波動對比，但基於步長 21 的 59 個區塊、path 1 |
| `path_volatility_check_by_ticker.png` | 277 KB | 同上，10 檔分開 |
| `path_distribution_check.png` | 132 KB | 分布對比，同樣基於步長 21、path 1 |
| `path_distribution_check_by_ticker.png` | 170 KB | 同上，10 檔分開 |

> `path_0001.csv` **就是**代表路徑：圖上那條實線畫的是它，所以不另外存一份 `pred_*.csv`。

### CSV 格式（長格式，每列一個 date × tic）

```
date,tic,close,high,low
2020-01-02,AAPL,72.617415,73.198157,71.106138
2020-01-02,AMGN,197.165490,198.862030,192.416500
```

1258 天 × 10 檔 = **12,580 列**。

| 欄位 | 原始定義 | 簡單報酬檔 | 股價檔 |
|---|---|---|---|
| `close` | `log(C_t / C_{t-1})` | `C_t/C_{t-1} − 1` | `C_0 × exp(cumsum(r_close))` |
| `high` | `log(H_t / C_t)` | `H_t/C_t − 1` | `C_t × exp(r_high)` |
| `low` | `log(L_t / C_t)` | `L_t/C_t − 1` | `C_t × exp(r_low)` |

> **`high` / `low` 不是日對日的報酬**，而是同一天的盤中高／低點相對於當天收盤價（所以 high ≥ 0、low ≤ 0）。它們**不能複利累加**，只能套在當天的收盤價上。

---

## 四、同名但不同的圖，怎麼分辨

四種分析圖在多個位置出現，**檔名不同、內容也不同**（全目錄 md5 掃描確認無重複）：

| 位置 | 前綴 | 窗口 | 情境 |
|---|---|---|---|
| 根目錄 | 無 | 步長 1，1238 個重疊窗口 | 無旋鈕 |
| `path_forecast/` | `path_` | 步長 21，59 個不重疊區塊 | 無旋鈕 |
| `path_forecast_knob-3/` | `path_` | 同上 | 旋鈕 −3 |
| `path_forecast_knob-3_w5/` | `path_` | 同上 | 旋鈕 −3 + guidance 5 |

**該看哪一個？**

- 要跟 MSE/MAE 對得上 → 根目錄那組（與 `test_*.csv` 同一套窗口）
- 要看波動與分布的真實性 → **`path_` 那組比較正確**。步長 1 的版本會把同一天重複計入 21 次，有效樣本數只有約 43（名目 1238）

---

## 五、兩個模型的差異

| | `_TWO`（x_start） | `_TWO_v`（v） |
|---|---|---|
| 網路預測的目標 | `x0` | `√ᾱ·ε − √(1−ᾱ)·x0` |
| test MSE | 1.1146 | 2.0848 |
| test MAE | 0.7093 | 1.0830 |
| 樣本間離散 / 真實波動 | 0.025 | **0.77** |
| 波動比 | 0.10 | **0.89** |

**MSE 變差是預期的。** MSE 獎勵條件均值，而 x_start 版本因為樣本塌縮，輸出的正好就是條件均值——它是用「不生成」換 MSE。v 每次抽的是分布中的一條樣本，單條當然離真實值更遠。

> 所以 v 之後不應該再用單樣本 MSE 當主要指標，要改用 CRPS 這類分布層面的指標。

---

## 六、重新產生

```bash
bash run_one.sh                                  # baseline，五步一次做完
bash run_one.sh v --parameterization v           # v 版

# 只重畫圖（純 CPU，讀快取，數秒）
python plot_test_analysis.py --setting <SETTING>

# 補跑某個情境的 500 條路徑
python predict_path.py <model flags> --path_knob -3 --knob_guidance 5 --sample_times 500
```
