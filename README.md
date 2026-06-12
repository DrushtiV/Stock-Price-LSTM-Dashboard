# Stock Price LSTM Dashboard 📈

A production-grade deep learning system for stock price prediction and forecasting,
built with a stacked LSTM + Monte-Carlo Dropout and served in a Streamlit dashboard.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Open **http://localhost:8501** → select a ticker → click **TRAIN & PREDICT**.

> **No internet?** Select `DEMO` or any built-in ticker. The dashboard automatically
> falls back to realistic synthetic data via Geometric Brownian Motion and runs the
> full LSTM pipeline identically.

---

## Project Structure
stock_lstm/

├── model.py          # Data fetch, feature engineering, LSTM, MC inference

├── dashboard.py      # Streamlit UI, all charts, session state

├── requirements.txt

└── README.md
---

## Architecture Overview
Yahoo Finance (yfinance)

│

▼

OHLCV DataFrame  ←─── GBM synthetic fallback if offline

│

▼

Feature Engineering (11 features)

Close, Open, High, Low

SMA-10, SMA-20, EMA-12

RSI-14, Bollinger width

Normalised Volume, Daily Return

│

▼

MinMaxScaler → [0, 1]

│

▼

Sliding Window  (seq_len=60, no shuffle)

X: (N, 60, 11)    y: (N,) next-day close

│

┌─────┴──────┐

Train (85%)   Test (15%)  ← chronological split, never shuffled

└─────┬──────┘

▼

LSTM(128) → Dropout(0.2)

LSTM(64)  → Dropout(0.2)

LSTM(32)  → Dropout(0.2)

Dense(16, relu)

Dense(1)

─────────────────────────

Optimizer : Adam (lr=1e-3)

Loss      : Huber

Callbacks : EarlyStopping(patience=10), ReduceLROnPlateau(patience=5)

│

▼

Monte-Carlo Dropout Inference

50 stochastic forward passes → mean + 95% CI

│

▼

Autoregressive Forecast (configurable, default 30 days)

Seed last 60 rows → predict next → append → drop oldest → repeat

│

▼

Streamlit Dashboard

KPI row: Last Close · RMSE · MAE · MAPE · N-day Forecast

Tab 1 : Candlestick + LSTM overlay + RSI + Volume

Tab 2 : N-day forecast with confidence bands + forecast table

Tab 3 : Loss curves, actual vs predicted scatter, residuals, model summary

Tab 4 : Raw OHLCV data table

---

## Technical Concepts

### 1. Sliding Window (Sequence Modelling)

An LSTM needs sequences, not individual rows. A sliding window of length 60
converts the time series into supervised learning samples:

```python
for i in range(60, len(data)):
    X.append(data[i-60:i])   # 60-day context window → features
    y.append(data[i, 0])     # predict next day's Close (scaled)
```

Each sample `X[i]` has shape `(60, 11)` — 60 time steps × 11 features.
The model learns temporal patterns across this entire window at once.

---

### 2. Chronological Train/Test Split — Never Shuffle

```python
split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]   # first 85% → train
```

Shuffling a time series before splitting leaks future information into the
training set (look-ahead bias). The split is strictly chronological: the model
is evaluated on data it has never seen, in the order it occurred.

---

### 3. Feature Engineering

11 features are computed from raw OHLCV data before scaling:

| Feature | Formula | Purpose |
|---------|---------|---------|
| Close, Open, High, Low | raw | price context |
| SMA-10, SMA-20 | rolling mean | trend direction |
| EMA-12 | exponential weighted | recent trend, reacts faster than SMA |
| RSI-14 | gain/loss ratio over 14 days | momentum / overbought/oversold |
| BB_width | `2σ / SMA-20` | volatility regime |
| Vol_norm | `Volume / 20-day avg volume` | relative buying pressure |
| Return | `pct_change(Close)` | daily momentum |

All 11 features are scaled together with a single `MinMaxScaler` so the model
receives inputs in the same `[0, 1]` range. The scaler is fit only on training
data to prevent test-set contamination.

---

### 4. Stacked LSTM Architecture
Input: (batch, 60, 11)

│

LSTM(128 units, return_sequences=True)    ← learns long-range dependencies

Dropout(0.2)

LSTM(64  units, return_sequences=True)    ← mid-level patterns

Dropout(0.2)

LSTM(32  units, return_sequences=False)   ← collapses sequence to a vector

Dropout(0.2)

Dense(16, activation='relu')             ← non-linear feature combination

Dense(1)                                 ← scalar next-day close prediction

**Why stacked?** Each LSTM layer learns increasingly abstract temporal
representations. The first captures short-term momentum; deeper layers pick up
weekly and monthly patterns.

**Why Dropout between layers?** Forces the network to learn redundant
representations, reducing overfitting on financial noise. In standard neural
networks dropout is disabled at inference — here it stays on, enabling
Monte-Carlo uncertainty estimation.

---

### 5. Huber Loss
L_δ(a) = { ½a²              if |a| ≤ δ

{ δ(|a| − ½δ)     otherwise

Behaves like MSE for small errors (smooth gradients) and MAE for large ones
(less penalty on outliers / price spikes). Ideal for financial time series
where earnings surprises produce single-day moves 5–10× the normal range.

---

### 6. Monte-Carlo Dropout (Uncertainty Quantification)

Standard neural networks produce a single point estimate. MC Dropout treats
the network as an approximate Bayesian model by keeping dropout layers **active
at inference time** and running multiple stochastic forward passes:

```python
preds = [model(X, training=True).numpy() for _ in range(50)]
# training=True keeps dropout ON → each pass has a different random mask

mean  = np.stack(preds).mean(axis=0)
lower = np.percentile(preds, 2.5,  axis=0)   # 95% confidence interval
upper = np.percentile(preds, 97.5, axis=0)
```

Each forward pass randomly deactivates 20% of neurons, producing a different
prediction. The spread across 50 passes reflects the model's epistemic
uncertainty — wider bands indicate the model is less confident (usually during
volatile regimes or near the edge of the training distribution).

**Reference:** Gal & Ghahramani (2016), *Dropout as a Bayesian Approximation*.

---

### 7. Autoregressive Future Forecast

No future features exist, so the model predicts one step at a time and feeds
its own output back as input:

```python
seed = scaled[-60:]          # last 60 known days (shape: 60×11)

for day in range(30):
    x    = seed[np.newaxis]  # shape: (1, 60, 11)
    pred = model(x, training=True).numpy()[0, 0]   # scaled close
    
    # Build next synthetic row:
    # use predicted close, carry all other features forward
    next_row    = seed[-1].copy()
    next_row[0] = pred
    seed        = np.vstack([seed[1:], next_row])   # slide window
```

Each prediction is run through MC Dropout (50 passes) so the confidence bands
widen as the horizon extends — the model is genuinely more uncertain about
day 30 than day 1.

---

### 8. Inverse Transform
All predictions are in the `[0, 1]` scaled space. To recover dollar prices:

```python
def inverse_close(scaler, scaled_values, n_features):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, 0] = scaled_values          # Close is column 0
    return scaler.inverse_transform(dummy)[:, 0]
```
A dummy array of the full feature width is required because `MinMaxScaler`
was fit on all 11 features together.

---

### 9. GBM Synthetic Fallback
When Yahoo Finance is unreachable (sandboxed environments, corporate proxies),
the dashboard generates realistic synthetic price data using **Geometric
Brownian Motion** with quarterly volatility regime shifts:
S(t) = S(t-1) × (1 + r),    r ~ N(μ, σ_t)
σ_t changes every ~60 trading days (quarterly regime shift)

Per-ticker seed parameters (`S₀`, `μ`, `σ`) are calibrated to approximate
real-world price levels and volatility for each symbol. The LSTM pipeline is
identical — only the data source differs. The UI shows a warning banner when
synthetic data is active.

---

## Evaluation Metrics
| Metric | Formula | Meaning |
|--------|---------|---------|
| RMSE | √(mean((actual − pred)²)) | Dollar error, penalises large misses heavily |
| MAE | mean(|actual − pred|) | Average dollar error, interpretable |
| MAPE | mean(|actual − pred| / actual) × 100 | % error, scale-independent |
| R² | 1 − SS_res / SS_tot | Variance explained (1 = perfect) |

---

## Hyperparameters Reference
| Parameter | Default | Tunable via UI |
|-----------|---------|---------------|
| Sequence length | 60 trading days | No |
| Forecast horizon | 30 days | Yes (5–90) |
| Training epochs | 50 | Yes (10–100) |
| MC Dropout samples | 50 | Yes (20–100) |
| History period | 2y | Yes (1y/2y/3y/5y) |
| Batch size | 32 | No |
| LSTM units | 128 → 64 → 32 | No |
| Dropout rate | 0.2 | No |
| Learning rate | 1e-3 | No (auto-reduced) |
| Train/test split | 85% / 15% | No |

---

## Limitations
- **Not financial advice.** This is a research/educational project demonstrating
  deep learning on time series. Do not use predictions for trading decisions.
- **Autoregressive error accumulation.** Each forecast step uses its own
  previous prediction. Errors compound; uncertainty bands reflect this.
- **No exogenous data.** The model has no access to news, earnings, macro
  indicators, or sentiment. Price-only models have hard upper bounds on accuracy.
- **Non-stationary markets.** A model trained on 2022–2024 may not generalise
  to regime shifts it has never seen.

---

## Requirements

yfinance>=0.2.38

tensorflow>=2.13.0

streamlit>=1.28.0

plotly>=5.18.0

pandas>=2.0.0

numpy>=1.24.0

scikit-learn>=1.3.0

Python 3.9+ recommended. Tested on CPU; GPU-compatible if TensorFlow-GPU is installed.

