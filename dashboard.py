"""
Stock Price LSTM Dashboard
==========================
Run with:  streamlit run dashboard.py
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta, datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model import (
    fetch_data, add_technical_features, preprocess,
    build_model, train_model,
    mc_predict, forecast_future, evaluate, inverse_close,
    SEQ_LEN, FORECAST_DAYS, MC_SAMPLES,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&display=swap');

html, body, [class*="css"] {
    background-color: #080c14 !important;
    color: #c9d1e0 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1320 !important;
    border-right: 1px solid #1a2235 !important;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

/* App background */
.stApp { background: #080c14 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #0d1320;
    border: 1px solid #1a2235;
    border-radius: 2px;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: #4a6080 !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.8rem !important;
    color: #e8f0ff !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* Buttons */
.stButton > button {
    background: #1a6fff !important;
    color: #fff !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Select / Slider */
.stSelectbox label, .stSlider label, .stTextInput label, .stNumberInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: #4a6080 !important;
    text-transform: uppercase !important;
}

/* Progress bar */
.stProgress > div > div { background: #1a6fff !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    color: #4a6080 !important;
}
.stTabs [aria-selected="true"] { color: #1a6fff !important; border-bottom-color: #1a6fff !important; }

/* Divider */
hr { border-color: #1a2235 !important; }

/* Info/warning boxes */
.stAlert { border-radius: 2px !important; }

/* Custom title */
.dash-title {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8f0ff;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 4px;
}
.dash-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4a6080;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 28px;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #4a6080;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2235;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    padding: 3px 10px;
    border: 1px solid #1a6fff;
    color: #1a6fff;
    text-transform: uppercase;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Color palette ─────────────────────────────────────────────────────────────
CLR = {
    "bg":       "#080c14",
    "surface":  "#0d1320",
    "border":   "#1a2235",
    "blue":     "#1a6fff",
    "green":    "#00d68f",
    "red":      "#ff4d6d",
    "orange":   "#ffb347",
    "text":     "#c9d1e0",
    "muted":    "#4a6080",
    "grid":     "#111824",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CLR["bg"],
    plot_bgcolor=CLR["surface"],
    font=dict(family="DM Mono, monospace", color=CLR["text"], size=11),
    margin=dict(l=16, r=16, t=32, b=16),
    xaxis=dict(
        gridcolor=CLR["grid"], showgrid=True,
        zeroline=False, linecolor=CLR["border"],
        tickfont=dict(size=10, color=CLR["muted"]),
    ),
    yaxis=dict(
        gridcolor=CLR["grid"], showgrid=True,
        zeroline=False, linecolor=CLR["border"],
        tickfont=dict(size=10, color=CLR["muted"]),
        tickprefix="$",
    ),
    legend=dict(
        bgcolor="rgba(13,19,32,0.8)", bordercolor=CLR["border"],
        borderwidth=1, font=dict(size=10),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="badge">Configuration</div>', unsafe_allow_html=True)
    st.markdown("### LSTM Stock Predictor")
    st.markdown('<div class="section-label">Ticker</div>', unsafe_allow_html=True)

    POPULAR = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "DEMO"]
    ticker_choice = st.selectbox("Select ticker", POPULAR, index=0)
    custom = st.text_input("Or enter custom ticker", placeholder="e.g. AMD, INTC")
    TICKER = custom.upper().strip() if custom.strip() else ticker_choice

    st.markdown('<div class="section-label" style="margin-top:16px">Data</div>', unsafe_allow_html=True)
    PERIOD = st.selectbox("History period", ["1y", "2y", "3y", "5y"], index=1)

    st.markdown('<div class="section-label" style="margin-top:16px">Model</div>', unsafe_allow_html=True)
    EPOCHS     = st.slider("Training epochs", 10, 100, 50, 5)
    FORECAST   = st.slider("Forecast days", 5, 90, 30, 5)
    MC_N       = st.slider("MC samples (confidence)", 20, 100, 50, 10)

    st.markdown("---")
    run_btn = st.button("🚀  TRAIN & PREDICT")
    st.markdown("---")
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:10px;color:#4a6080;">'
        'LSTM · Monte-Carlo Dropout<br>11 engineered features<br>60-day sliding window<br>'
        'Chronological train/test split</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown(
        f'<div class="dash-title">{TICKER} <span style="color:#1a6fff">Price</span> Predictor</div>'
        '<div class="dash-subtitle">LSTM · Deep Learning · Time Series Forecasting</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (sandbox fallback / DEMO mode)
# ─────────────────────────────────────────────────────────────────────────────

TICKER_SEEDS = {
    "AAPL": (185, 0.00035, 0.014),
    "MSFT": (380, 0.00040, 0.013),
    "GOOGL":(175, 0.00030, 0.015),
    "AMZN": (195, 0.00038, 0.016),
    "TSLA": (250, 0.00020, 0.025),
    "NVDA": (900, 0.00060, 0.022),
    "META": (530, 0.00045, 0.016),
    "NFLX": (720, 0.00025, 0.017),
    "DEMO": (150, 0.00030, 0.018),
}

def generate_synthetic(ticker: str, period: str = "2y") -> tuple[pd.DataFrame, bool]:
    """
    Try Yahoo Finance first; fall back to realistic GBM synthetic data.
    Returns (df, is_synthetic).
    """
    try:
        df = fetch_data(ticker, period)
        if len(df) > 100:
            return df, False
    except Exception:
        pass

    # ── Synthetic fallback ──
    period_days = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}.get(period, 504)
    params = TICKER_SEEDS.get(ticker, (150, 0.0003, 0.016))
    S0, mu, sigma = params

    seed_val = sum(ord(c) for c in ticker) % 1000
    rng  = np.random.default_rng(seed_val)
    n    = period_days
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    # GBM with mild mean-reversion + regime changes
    prices = [S0]
    vol = sigma
    for i in range(1, n):
        if i % 60 == 0:                          # regime shock every ~quarter
            vol = sigma * rng.uniform(0.7, 1.4)
        r = rng.normal(mu, vol)
        prices.append(prices[-1] * (1 + r))

    prices = np.array(prices)
    close  = prices
    open_  = np.roll(close, 1); open_[0] = S0
    noise  = lambda s: np.abs(rng.normal(0, s, n))
    high   = np.maximum(close, open_) * (1 + noise(0.006))
    low    = np.minimum(close, open_) * (1 - noise(0.006))
    volume = rng.lognormal(20, 0.35, n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df, True


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_main_chart(df, data, model, eval_res, forecast, forecast_dates, ticker):
    """Candlestick + LSTM overlay + confidence bands + forecast."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.20, 0.18],
        vertical_spacing=0.02,
        subplot_titles=("", "", ""),
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color=CLR["green"], width=1), fillcolor=CLR["green"]),
        decreasing=dict(line=dict(color=CLR["red"],   width=1), fillcolor=CLR["red"]),
        showlegend=True,
    ), row=1, col=1)

    # ── SMAs ─────────────────────────────────────────────────────────────────
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"],
            name="SMA 20", line=dict(color=CLR["orange"], width=1, dash="dot"),
            opacity=0.7,
        ), row=1, col=1)
    if "SMA_10" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_10"],
            name="SMA 10", line=dict(color="#a78bfa", width=1, dash="dot"),
            opacity=0.7,
        ), row=1, col=1)

    # ── Test predictions + confidence band ───────────────────────────────────
    pred_mean_s, pred_low_s, pred_high_s = mc_predict(model, data["X_test"], n_samples=MC_N)
    n_feat = data["n_features"]
    sc     = data["scaler"]
    pred_r = inverse_close(sc, pred_mean_s, n_feat)
    low_r  = inverse_close(sc, pred_low_s,  n_feat)
    high_r = inverse_close(sc, pred_high_s, n_feat)
    test_dates = data["test_dates"]

    fig.add_trace(go.Scatter(
        x=list(test_dates) + list(reversed(test_dates)),
        y=list(high_r) + list(reversed(low_r)),
        fill="toself",
        fillcolor="rgba(26,111,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=test_dates, y=pred_r,
        name="LSTM Prediction",
        line=dict(color=CLR["blue"], width=2),
    ), row=1, col=1)

    # ── Future forecast ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(reversed(forecast_dates)),
        y=list(forecast["upper"]) + list(reversed(forecast["lower"])),
        fill="toself",
        fillcolor="rgba(0,214,143,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Forecast CI",
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast["mean"],
        name="Forecast",
        line=dict(color=CLR["green"], width=2, dash="dash"),
    ), row=1, col=1)

    # Vertical line at forecast start
    fig.add_vline(
        x=str(df.index[-1]), line_dash="dot",
        line_color=CLR["muted"], line_width=1, row=1, col=1,
    )

    # ── RSI ───────────────────────────────────────────────────────────────────
    if "RSI_14" in df.columns:
        rsi = df["RSI_14"]
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            name="RSI 14", line=dict(color=CLR["orange"], width=1.5),
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=CLR["red"],   line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=CLR["green"], line_width=0.8, row=2, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    colors_v = [CLR["green"] if c >= o else CLR["red"]
                for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors_v, opacity=0.6,
        showlegend=False,
    ), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = dict(**PLOTLY_LAYOUT)
    layout.update(
        height=700,
        title=dict(
            text=f"<b>{ticker}</b> — LSTM Deep Learning Forecast",
            font=dict(family="Fraunces, serif", size=16, color=CLR["text"]),
            x=0.01,
        ),
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_layout(**layout)

    # Y-axis labels
    fig.update_yaxes(title_text="Price (USD)", tickprefix="$", row=1, col=1,
                     gridcolor=CLR["grid"], title_font=dict(size=9, color=CLR["muted"]))
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1,
                     gridcolor=CLR["grid"], title_font=dict(size=9, color=CLR["muted"]))
    fig.update_yaxes(title_text="Volume", row=3, col=1,
                     gridcolor=CLR["grid"], title_font=dict(size=9, color=CLR["muted"]))

    return fig


def build_loss_chart(history):
    fig = go.Figure()
    epochs_range = list(range(1, len(history.history["loss"]) + 1))
    fig.add_trace(go.Scatter(
        x=epochs_range, y=history.history["loss"],
        name="Train Loss", line=dict(color=CLR["blue"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs_range, y=history.history["val_loss"],
        name="Val Loss", line=dict(color=CLR["orange"], width=2, dash="dash"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title=dict(text="Training Loss (Huber)", font=dict(size=13), x=0.01),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Epoch"),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Loss", tickprefix=""),
    )
    return fig


def build_scatter_chart(eval_res):
    actual = eval_res["actual_real"]
    pred   = eval_res["pred_real"]
    mn, mx = min(actual.min(), pred.min()), max(actual.max(), pred.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=pred,
        mode="markers",
        marker=dict(color=CLR["blue"], size=4, opacity=0.6),
        name="Actual vs Predicted",
    ))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        line=dict(color=CLR["muted"], dash="dash", width=1),
        name="Perfect fit",
        showlegend=True,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title=dict(text="Actual vs Predicted (Test Set)", font=dict(size=13), x=0.01),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Actual ($)", tickprefix="$"),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Predicted ($)", tickprefix="$"),
    )
    return fig


def build_residual_chart(eval_res, test_dates):
    residuals = eval_res["actual_real"] - eval_res["pred_real"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=test_dates, y=residuals,
        marker_color=[CLR["green"] if r >= 0 else CLR["red"] for r in residuals],
        opacity=0.7, name="Residual",
    ))
    fig.add_hline(y=0, line_color=CLR["muted"], line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title=dict(text="Prediction Residuals (Actual − Predicted)", font=dict(size=13), x=0.01),
        xaxis=dict(**PLOTLY_LAYOUT["xaxis"]),
        yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Residual ($)", tickprefix="$"),
    )
    return fig


def build_forecast_chart(forecast, forecast_dates, last_price):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(reversed(forecast_dates)),
        y=list(forecast["upper"]) + list(reversed(forecast["lower"])),
        fill="toself",
        fillcolor="rgba(0,214,143,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast["mean"],
        line=dict(color=CLR["green"], width=2.5),
        name="Mean Forecast",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast["upper"],
        line=dict(color=CLR["green"], width=1, dash="dot"),
        name="Upper 95%",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast["lower"],
        line=dict(color=CLR["red"], width=1, dash="dot"),
        name="Lower 95%",
    ))
    fig.add_hline(y=last_price, line_dash="dot", line_color=CLR["muted"],
                  annotation_text=f"Last close ${last_price:.2f}",
                  annotation_font=dict(color=CLR["muted"], size=10))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=360,
        title=dict(
            text=f"<b>{len(forecast['mean'])}-Day Forward Forecast</b> with MC Dropout Confidence Bands",
            font=dict(size=14), x=0.01,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────────────────────

if "trained" not in st.session_state:
    st.session_state.trained = False

# Auto-run if first load with DEMO ticker
if TICKER == "DEMO" and not st.session_state.trained:
    run_btn = True

if run_btn:
    st.session_state.trained = False

    with st.status("🔄 Loading data…", expanded=True) as status:
        st.write(f"Fetching {TICKER} ({PERIOD})…")
        raw_df, is_synthetic = generate_synthetic(TICKER, PERIOD)

        if is_synthetic:
            st.warning(
                f"⚠️ Could not reach Yahoo Finance (network restriction). "
                f"Displaying **realistic synthetic** price data for {TICKER} "
                f"generated with Geometric Brownian Motion. The LSTM pipeline "
                f"is fully operational — swap in real data when network permits.",
                icon="🔶",
            )

        st.write("Engineering features…")
        df = add_technical_features(raw_df)

        st.write("Building sequences & scaling…")
        data = preprocess(df, seq_len=SEQ_LEN)

        st.write("Constructing LSTM architecture…")
        model = build_model(SEQ_LEN, data["n_features"])

        st.write(f"Training for up to {EPOCHS} epochs (early stopping active)…")
        progress = st.progress(0)

        class ProgressCallback:
            def __init__(self, total, bar): self.total, self.bar = total, bar
            def on_epoch_end(self, epoch, logs=None):
                self.bar.progress(min((epoch + 1) / self.total, 1.0))

        import tensorflow as tf
        cb = ProgressCallback(EPOCHS, progress)

        class _CB(tf.keras.callbacks.Callback):
            def __init__(self, tracker): self.tracker = tracker
            def on_epoch_end(self, epoch, logs=None): self.tracker.on_epoch_end(epoch, logs)

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        history = model.fit(
            data["X_train"], data["y_train"],
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=32,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
                _CB(cb),
            ],
            shuffle=False,
            verbose=0,
        )
        progress.progress(1.0)

        st.write(f"Evaluating on test set…")
        eval_res = evaluate(model, data)

        st.write(f"Forecasting {FORECAST} days ahead with MC Dropout…")
        forecast = forecast_future(
            model, data["scaler"], data["scaled"],
            data["n_features"], days=FORECAST, n_samples=MC_N,
        )
        last_date = df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=FORECAST)

        status.update(label="✅ Done!", state="complete", expanded=False)

    # Store in session
    st.session_state.update({
        "trained": True, "df": df, "raw_df": raw_df,
        "data": data, "model": model, "history": history,
        "eval_res": eval_res, "forecast": forecast,
        "forecast_dates": forecast_dates,
        "ticker": TICKER, "is_synthetic": is_synthetic,
    })


# ─────────────────────────────────────────────────────────────────────────────
# RENDER RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.trained:
    df           = st.session_state.df
    raw_df       = st.session_state.raw_df
    data         = st.session_state.data
    model        = st.session_state.model
    history      = st.session_state.history
    eval_res     = st.session_state.eval_res
    forecast     = st.session_state.forecast
    forecast_dates = st.session_state.forecast_dates
    ticker       = st.session_state.ticker
    is_synthetic = st.session_state.is_synthetic

    last_price   = float(df["Close"].iloc[-1])
    prev_price   = float(df["Close"].iloc[-2])
    pct_chg      = (last_price - prev_price) / prev_price * 100
    forecast_end = float(forecast["mean"][-1])
    forecast_pct = (forecast_end - last_price) / last_price * 100

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Last Close",    f"${last_price:.2f}",    f"{pct_chg:+.2f}%")
    k2.metric("RMSE",          f"${eval_res['RMSE']:.2f}")
    k3.metric("MAE",           f"${eval_res['MAE']:.2f}")
    k4.metric("MAPE",          f"{eval_res['MAPE']:.2f}%")
    k5.metric(f"{len(forecast['mean'])}d Forecast", f"${forecast_end:.2f}", f"{forecast_pct:+.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Main Chart",
        "🔮  Forecast",
        "🧪  Model Analysis",
        "📋  Raw Data",
    ])

    with tab1:
        st.markdown('<div class="section-label">Price History · LSTM Overlay · Indicators</div>', unsafe_allow_html=True)
        fig_main = build_main_chart(df, data, model, eval_res, forecast, forecast_dates, ticker)
        st.plotly_chart(fig_main, use_container_width=True)

        info_cols = st.columns(4)
        info_cols[0].metric("Training samples", f"{len(data['X_train']):,}")
        info_cols[1].metric("Test samples",      f"{len(data['X_test']):,}")
        info_cols[2].metric("Features",          data["n_features"])
        info_cols[3].metric("R² Score",          f"{eval_res['R²']:.4f}")

    with tab2:
        st.markdown('<div class="section-label">Forward Forecast with Monte-Carlo Confidence Bands</div>', unsafe_allow_html=True)
        fig_fc = build_forecast_chart(forecast, forecast_dates, last_price)
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table
        fc_df = pd.DataFrame({
            "Date":       forecast_dates.strftime("%Y-%m-%d"),
            "Forecast":   [f"${v:.2f}" for v in forecast["mean"]],
            "Lower 95%":  [f"${v:.2f}" for v in forecast["lower"]],
            "Upper 95%":  [f"${v:.2f}" for v in forecast["upper"]],
            "Change":     [f"{((v - last_price)/last_price*100):+.2f}%" for v in forecast["mean"]],
        })
        st.dataframe(fc_df, use_container_width=True, height=300,
                     hide_index=True)

        st.info(
            "**Confidence bands** are generated using Monte-Carlo Dropout: "
            f"{MC_N} stochastic forward passes with dropout layers kept ON at inference. "
            "Wider bands = higher model uncertainty. Uncertainty grows with forecast horizon.",
            icon="ℹ️",
        )

    with tab3:
        st.markdown('<div class="section-label">Training Diagnostics</div>', unsafe_allow_html=True)
        col_l, col_s = st.columns(2)
        with col_l:
            st.plotly_chart(build_loss_chart(history), use_container_width=True)
        with col_s:
            st.plotly_chart(build_scatter_chart(eval_res), use_container_width=True)

        st.plotly_chart(build_residual_chart(eval_res, data["test_dates"]), use_container_width=True)

        with st.expander("Model Architecture"):
            lines = []
            model.summary(print_fn=lambda x: lines.append(x))
            st.code("\n".join(lines), language="text")

        with st.expander("Feature Columns Used"):
            st.write(data["feature_cols"])

        with st.expander("Hyperparameters"):
            st.json({
                "sequence_length": SEQ_LEN,
                "forecast_days": FORECAST,
                "mc_dropout_samples": MC_N,
                "epochs_run": len(history.history["loss"]),
                "batch_size": 32,
                "optimizer": "Adam",
                "loss_function": "Huber",
                "architecture": "LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)",
                "train_test_split": "Chronological (no shuffle)",
                "scaler": "MinMaxScaler [0, 1]",
            })

    with tab4:
        st.markdown('<div class="section-label">Historical OHLCV Data</div>', unsafe_allow_html=True)
        display_df = df[["Open","High","Low","Close","Volume"]].copy()
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        display_df = display_df.sort_index(ascending=False)
        for col in ["Open","High","Low","Close"]:
            display_df[col] = display_df[col].map("${:.2f}".format)
        display_df["Volume"] = display_df["Volume"].map("{:,.0f}".format)
        st.dataframe(display_df, use_container_width=True, height=500)

        if is_synthetic:
            st.warning("⚠️ Data shown is synthetic (GBM simulation). Yahoo Finance was unreachable.", icon="🔶")

else:
    # Welcome screen
    st.markdown("""
    <div style="
        border: 1px solid #1a2235;
        background: #0d1320;
        padding: 48px;
        text-align: center;
        margin: 40px 0;
    ">
        <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:3px;color:#4a6080;margin-bottom:16px;">
            READY
        </div>
        <div style="font-family:'Fraunces',serif;font-size:2rem;color:#e8f0ff;margin-bottom:12px;">
            Configure & Train
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#4a6080;max-width:480px;margin:0 auto;">
            Select a ticker and period in the sidebar, then click
            <span style="color:#1a6fff;">TRAIN &amp; PREDICT</span> to run the full
            LSTM pipeline — feature engineering, training, MC-Dropout confidence
            bands, and 30-day forward forecast.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    def info_card(col, icon, title, body):
        col.markdown(f"""
        <div style="border:1px solid #1a2235;background:#0d1320;padding:20px;">
            <div style="font-size:1.4rem;margin-bottom:8px;">{icon}</div>
            <div style="font-family:'Fraunces',serif;font-size:1rem;color:#e8f0ff;margin-bottom:6px;">{title}</div>
            <div style="font-family:'DM Mono',monospace;font-size:11px;color:#4a6080;line-height:1.6;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    info_card(c1, "🧠", "Stacked LSTM",
              "Three LSTM layers (128→64→32 units) with MC-Dropout. "
              "Trained end-to-end with Huber loss for outlier robustness.")
    info_card(c2, "📐", "11 Features",
              "Close, OHLC, SMA-10/20, EMA-12, RSI-14, Bollinger width, "
              "normalised volume, daily return — all scaled to [0,1].")
    info_card(c3, "🔮", "Uncertainty Bands",
              "Monte-Carlo Dropout generates 50 stochastic predictions per step "
              "for 95% confidence intervals on both test-set and future forecasts.")
