# Stock Price LSTM Dashboard 📈

A production-grade deep learning system for stock price prediction and forecasting, built with a stacked LSTM + Monte-Carlo Dropout and served in a Streamlit dashboard.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Open **http://localhost:8501** → select a ticker → click **TRAIN & PREDICT**.

---

## Architecture

<img width="2720" height="3680" alt="lstm_stock_predictor_pipeline" src="https://github.com/user-attachments/assets/48da3dba-5b2a-4c2d-adcf-a8e5777697ff" />

```
Yahoo Finance (yfinance)
        │
        ▼
  OHLCV DataFrame
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
   Train (85%)   Test (15%)  ← chronological split
  └─────┬──────┘
        ▼
  LSTM(128) → Dropout(0.2)
  LSTM(64)  → Dropout(0.2)
  LSTM(32)  → Dropout(0.2)
  Dense(16, relu)
  Dense(1)
  ─────────────────────────
  Optimizer: Adam (lr=1e-3)
  Loss:      Huber
  Callbacks: EarlyStopping, ReduceLROnPlateau
        │
        ▼
Monte-Carlo Dropout Inference
  50 stochastic forward passes → mean + 95% CI
        │
        ▼
Autoregressive Forecast (30 days)
  Seed last 60 days → predict next → append → drop oldest → repeat
        │
        ▼
Streamlit Dashboard
  Tab 1: Candlestick + LSTM overlay + RSI + Volume
  Tab 2: 30-day forecast with confidence bands
  Tab 3: Loss curves, scatter, residuals, model summary
  Tab 4: Raw OHLCV data table
```

---

## Key Technical Concepts

### Sliding Window (Sequence Modelling)
```python
for i in range(60, len(data)):
    X.append(data[i-60:i])   # 60-day window → features
    y.append(data[i, 0])     # next day Close (scaled)
```

### No-Shuffle Time-Series Split
```python
split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]  # chronological!
```
Shuffling leaks future data into training — never do this for time series.

### MinMaxScaler + Inverse Transform
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(raw_features)
# After prediction:
pred_real = scaler.inverse_transform(dummy)[:, 0]  # Close is column 0
```

### Monte-Carlo Dropout Confidence Bands
```python
# Dropout kept ON at inference (training=True)
preds = [model(X, training=True) for _ in range(50)]
mean  = np.stack(preds).mean(axis=0)
lower = np.percentile(preds, 2.5,  axis=0)
upper = np.percentile(preds, 97.5, axis=0)
```

### Autoregressive Future Forecast
```python
seed = scaled[-60:]  # last known 60 days
for day in range(30):
    pred = model(seed[np.newaxis], training=True)
    seed = np.vstack([seed[1:], update_row(pred)])
```

### Huber Loss (Outlier Robustness)
Behaves like MSE for small errors, MAE for large ones — ideal for price series with occasional spikes.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE   | Root Mean Squared Error — penalises large errors |
| MAE    | Mean Absolute Error — average dollar error |
| MAPE   | Mean Absolute Percentage Error |
| R²     | Coefficient of determination (1 = perfect) |

---

## Network Fallback

If Yahoo Finance is unreachable (e.g., corporate proxy, sandboxed environment), the dashboard automatically generates realistic synthetic price data using **Geometric Brownian Motion** with quarterly volatility regime shifts.
The LSTM pipeline is identical — only the data source differs.
To force real data, run in an environment with internet access to `finance.yahoo.com`.

---

## Project Structure

```
stock_lstm/
├── model.py        # Data fetch, feature engineering, LSTM, MC inference
├── dashboard.py    # Streamlit UI, all charts, session state
├── requirements.txt
└── README.md
```
