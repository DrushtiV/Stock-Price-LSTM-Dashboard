"""
LSTM Stock Price Model
======================
Handles data fetching, preprocessing, model building, training,
and forward prediction with confidence bands.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")


# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN      = 60    # sliding window: 60 trading days → predict next
FORECAST_DAYS = 30   # how many days ahead to forecast
EPOCHS       = 60
BATCH_SIZE   = 32
TEST_RATIO   = 0.15  # last 15% of data is test set (no shuffle!)
MC_SAMPLES   = 50    # Monte-Carlo dropout passes for confidence bands


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling technical indicators as extra model inputs."""
    d = df.copy()
    close = d["Close"]

    # Simple Moving Averages
    d["SMA_10"]  = close.rolling(10).mean()
    d["SMA_20"]  = close.rolling(20).mean()

    # Exponential Moving Average
    d["EMA_12"]  = close.ewm(span=12, adjust=False).mean()

    # RSI (14-period)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Band width
    sma20   = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    d["BB_width"] = (2 * std20) / (sma20 + 1e-9)

    # Normalised volume
    d["Vol_norm"] = d["Volume"] / d["Volume"].rolling(20).mean()

    # Daily return
    d["Return"] = close.pct_change()

    return d.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(data: np.ndarray, seq_len: int):
    """Sliding-window sequence builder.  X shape: (N, seq_len, features)"""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])           # index 0 = Close (scaled)
    return np.array(X), np.array(y)


def preprocess(df: pd.DataFrame, seq_len: int = SEQ_LEN, test_ratio: float = TEST_RATIO):
    """
    Scale features, build sequences, split train/test (chronological).
    Returns a dict with everything the dashboard and trainer need.
    """
    feature_cols = [
        "Close", "Open", "High", "Low",
        "SMA_10", "SMA_20", "EMA_12", "RSI_14",
        "BB_width", "Vol_norm", "Return",
    ]
    # Only use columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    raw = df[feature_cols].values.astype(np.float32)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(raw)

    X, y = build_sequences(scaled, seq_len)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Dates aligned to sequence targets (offset by seq_len)
    dates = df.index[seq_len:]
    train_dates = dates[:split]
    test_dates  = dates[split:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "train_dates": train_dates, "test_dates": test_dates,
        "scaler": scaler,
        "scaled": scaled,
        "raw":    raw,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    """
    Stacked LSTM with MC-Dropout.
    Dropout stays ON at inference for Monte-Carlo confidence bands.
    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ], name="StockLSTM")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="huber",          # robust to price spikes
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, data: dict, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Train with early stopping + LR reduction."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,   # ← critical: time-series must NOT be shuffled
        verbose=0,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 6. INVERSE-TRANSFORM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def inverse_close(scaler: MinMaxScaler, scaled_values: np.ndarray, n_features: int) -> np.ndarray:
    """Inverse-transform only the Close column (index 0)."""
    dummy = np.zeros((len(scaled_values), n_features), dtype=np.float32)
    dummy[:, 0] = scaled_values.ravel()
    return scaler.inverse_transform(dummy)[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# 7. INFERENCE + MONTE-CARLO CONFIDENCE BANDS
# ─────────────────────────────────────────────────────────────────────────────

def mc_predict(model, X: np.ndarray, n_samples: int = MC_SAMPLES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run MC-Dropout: forward-pass n_samples times with dropout ON.
    Returns (mean, lower_95, upper_95) in scaled space.
    """
    preds = np.stack(
        [model(X, training=True).numpy().ravel() for _ in range(n_samples)],
        axis=0,
    )
    mean  = preds.mean(axis=0)
    lower = np.percentile(preds, 2.5,  axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return mean, lower, upper


def forecast_future(
    model,
    scaler: MinMaxScaler,
    scaled: np.ndarray,
    n_features: int,
    seq_len: int = SEQ_LEN,
    days: int     = FORECAST_DAYS,
    n_samples: int = MC_SAMPLES,
) -> dict:
    """
    Autoregressively forecast `days` ahead.
    Each step: append prediction, drop oldest, repeat.
    Returns dict with arrays for mean/low/high in real price space.
    """
    # Seed sequence: last seq_len rows of scaled data
    seed = scaled[-seq_len:].copy()          # (seq_len, n_features)

    all_preds = []
    for _ in range(days):
        x    = seed[np.newaxis, :, :]         # (1, seq_len, n_features)
        preds = np.array([model(x, training=True).numpy()[0, 0] for _ in range(n_samples)])
        all_preds.append(preds)

        # Build next row: use predicted close, carry forward other features
        next_row        = seed[-1].copy()
        next_row[0]     = preds.mean()
        seed            = np.vstack([seed[1:], next_row[np.newaxis, :]])

    all_preds  = np.array(all_preds)          # (days, n_samples)
    mean_s     = all_preds.mean(axis=1)
    lower_s    = np.percentile(all_preds, 2.5,  axis=1)
    upper_s    = np.percentile(all_preds, 97.5, axis=1)

    return {
        "mean":  inverse_close(scaler, mean_s,  n_features),
        "lower": inverse_close(scaler, lower_s, n_features),
        "upper": inverse_close(scaler, upper_s, n_features),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, data: dict) -> dict:
    """Compute RMSE, MAE, MAPE on the held-out test set."""
    scaler     = data["scaler"]
    n_features = data["n_features"]

    pred_s, _, _ = mc_predict(model, data["X_test"])
    actual_s     = data["y_test"]

    pred_real   = inverse_close(scaler, pred_s,    n_features)
    actual_real = inverse_close(scaler, actual_s,  n_features)

    rmse  = np.sqrt(mean_squared_error(actual_real, pred_real))
    mae   = mean_absolute_error(actual_real, pred_real)
    mape  = np.mean(np.abs((actual_real - pred_real) / (actual_real + 1e-9))) * 100
    r2    = 1 - np.sum((actual_real - pred_real) ** 2) / np.sum((actual_real - actual_real.mean()) ** 2)

    return {
        "RMSE":  round(float(rmse),  3),
        "MAE":   round(float(mae),   3),
        "MAPE":  round(float(mape),  3),
        "R²":    round(float(r2),    4),
        "pred_real":   pred_real,
        "actual_real": actual_real,
    }
