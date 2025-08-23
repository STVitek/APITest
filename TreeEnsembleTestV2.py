"""
Hybrid XGBoost + LightGBM Walk‑Forward Stock Classifier

Goal
-----
Predict whether tomorrow's close will be higher than today's close (binary classification)
using an ensemble of XGBoost + LightGBM, trained in a realistic walk‑forward fashion.

Highlights (where each model’s strengths are leveraged)
-------------------------------------------------------
- XGBoost: robust/stable on noisy tabular features; good first baseline.
- LightGBM: fast on large histories; native categorical handling (weekday/month);
            deeper leaf‑wise splits can capture feature interactions.
- Ensemble: weighted by time‑series CV performance inside each training window.

Usage
-----
1) Ensure dependencies are installed: pip install pandas numpy scikit-learn xgboost lightgbm yfinance
2) Run this file directly to fetch a ticker via yfinance (default: 'AAPL').
   Or import and call `run_experiment(df)` with your own OHLCV DataFrame.

Notes
-----
- This script is intentionally transparent and verbose for learning purposes.
- All feature engineering is done with pandas/numpy (no external TA deps).
- Walk‑forward uses an expanding window by default; set `rolling_window_days` for a fixed window.
- For speed, models are re‑fit every `refit_every_n_days` (default 5 trading days).
- No transaction cost modeling included here; you can extend in the signal->strategy section.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

# Optional data download for demo
try:
    import yfinance as yf
except Exception:
    yf = None

# Plotting import
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Feature Engineering (pure pandas/numpy)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's RSI
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'])
        out = out.set_index('Date').sort_index()
    out = out.sort_index()

    # --- Returns ---
    out['ret_1d'] = out['Close'].pct_change()
    out['log_ret_1d'] = np.log(out['Close']).diff()

    # Lag returns so we only use past info
    for lag in [1, 2, 3, 5, 10, 21]:
        out[f'ret_lag_{lag}'] = out['ret_1d'].shift(lag)

    # Rolling stats (shifted by 1 to avoid lookahead)
    for win in [5, 10, 21, 63]:
        roll_mean = out['ret_1d'].rolling(win).mean()
        roll_std = out['ret_1d'].rolling(win).std()
        zscore = (out['Close'] - out['Close'].rolling(win).mean()) / (out['Close'].rolling(win).std() + 1e-12)

        out[f'roll_mean_{win}'] = roll_mean.shift(1)
        out[f'roll_std_{win}'] = roll_std.shift(1)
        out[f'close_z_{win}'] = zscore.shift(1)

    # Volatility proxies (shifted)
    out['hl_range'] = ((out['High'] - out['Low']) / out['Close'].shift(1)).shift(1)
    out['vol_park_21'] = (np.log(out['High']) - np.log(out['Low'])).rolling(21).std().shift(1)

    # Volume features (shifted)
    out['vol_chg'] = out['Volume'].pct_change().shift(1)
    for win in [5, 21, 63]:
        vol_z = (out['Volume'] - out['Volume'].rolling(win).mean()) / (out['Volume'].rolling(win).std() + 1e-12)
        out[f'vol_z_{win}'] = vol_z.shift(1)

    # TA indicators (lagged)
    out['rsi_14'] = _rsi(out['Close'], 14).shift(1)
    macd_line, signal_line = _macd(out['Close'])
    out['macd_line'] = macd_line.shift(1)
    out['macd_signal'] = signal_line.shift(1)
    out['macd_hist'] = (macd_line - signal_line).shift(1)

    # Calendar categorical features
    out['dow'] = out.index.dayofweek.astype('int16')
    out['month'] = out.index.month.astype('int16')

    # --- Target: Will today's close > today's open? ---
    out['y'] = (out['Close'] > out['Open']).astype('int8')

    out = out.dropna()
    return out


# ---------------------------------------------------------------------------
# Time‑Series CV inside each train window (to set ensemble weights)
# ---------------------------------------------------------------------------

def ts_cv_weights(X: pd.DataFrame, y: pd.Series,
                  xgb_params: dict, lgbm_params: dict,
                  n_splits: int = 3) -> Tuple[float, float]:
    """Compute ROC AUC via TimeSeriesSplit and convert to weights.
    We use ROC AUC because probabilities are useful; switch to accuracy if preferred.
    Returns (w_xgb, w_lgbm) summing to 1.0.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    auc_xgb, auc_lgbm = [], []

    for tr_idx, va_idx in splitter.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        xgb = XGBClassifier(**xgb_params)
        lgb = LGBMClassifier(**lgbm_params)

        # LightGBM can accept categorical features directly; cast here
        X_tr_lgb = X_tr.copy()
        X_va_lgb = X_va.copy()
        for cat in ['dow', 'month']:
            if cat in X_tr_lgb.columns:
                X_tr_lgb[cat] = X_tr_lgb[cat].astype('category')
                X_va_lgb[cat] = X_va_lgb[cat].astype('category')

        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb.fit(X_tr_lgb, y_tr, eval_set=[(X_va_lgb, y_va)])  # <-- Remove verbose=False here

        p_xgb = xgb.predict_proba(X_va)[:, 1]
        p_lgb = lgb.predict_proba(X_va_lgb)[:, 1]

        try:
            auc_xgb.append(roc_auc_score(y_va, p_xgb))
        except Exception:
            auc_xgb.append(0.5)
        try:
            auc_lgbm.append(roc_auc_score(y_va, p_lgb))
        except Exception:
            auc_lgbm.append(0.5)

    mean_xgb = float(np.nanmean(auc_xgb))
    mean_lgb = float(np.nanmean(auc_lgbm))

    # Convert to weights; add small epsilon to avoid zero division
    eps = 1e-6
    total = mean_xgb + mean_lgb + eps
    w_xgb = (mean_xgb + eps) / total
    w_lgb = (mean_lgb + eps) / total
    return w_xgb, w_lgb


# ---------------------------------------------------------------------------
# Walk‑Forward Backtest / Prediction
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    start_train_days: int = 504        # ~2 trading years
    refit_every_n_days: int = 5        # refit weekly for speed
    rolling_window_days: Optional[int] = 504  # if set, use fixed-size rolling window; else expanding
    cv_splits: int = 3


@dataclass
class ModelParams:
    xgb_params: Dict
    lgbm_params: Dict


def walk_forward_ensemble(df_feat: pd.DataFrame,
                          cfg: WalkForwardConfig,
                          params: ModelParams,
                          feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Run a walk‑forward expanding/rolling window.
    Returns a DataFrame with columns: ['pred_xgb','pred_lgb','pred_ens','y','Close'] indexed by date.
    """
    if feature_cols is None:
        # Exclude target and obvious non-features
        feature_cols = [c for c in df_feat.columns if c not in ('y',)]

    dates = df_feat.index
    results = []

    # Pre‑cast for LightGBM categoricals
    cat_cols = [c for c in ['dow','month'] if c in feature_cols]

    for i in range(cfg.start_train_days, len(df_feat)-1):
        # Skip unless it's refit day or the very first prediction point
        if (i - cfg.start_train_days) % cfg.refit_every_n_days != 0 and len(results) > 0:
            # Reuse last models/weights if not refitting (faster). Predict only for date i.
            row = df_feat.iloc[[i]]
            X_row = row[feature_cols]

            # LGB categorical casting for a single row
            X_row_lgb = X_row.copy()
            for c in cat_cols:
                X_row_lgb[c] = X_row_lgb[c].astype('category')

            p_xgb = xgb.predict_proba(X_row)[:, 1][0]
            p_lgb = lgb.predict_proba(X_row_lgb)[:, 1][0]
            p_ens = w_xgb * p_xgb + w_lgb * p_lgb

            results.append({
                'Date': dates[i],
                'pred_xgb': p_xgb,
                'pred_lgb': p_lgb,
                'pred_ens': p_ens,
                'y': int(df_feat['y'].iloc[i]),
                'Close': float(df_feat['Close'].iloc[i])
            })
            continue

        # Define train window
        start_idx = 0 if cfg.rolling_window_days is None else max(0, i - cfg.rolling_window_days)
        tr = df_feat.iloc[start_idx:i]
        X_tr = tr[feature_cols]
        y_tr = tr['y']

        # Compute model weights via time‑series CV inside the training window
        w_xgb, w_lgb = ts_cv_weights(X_tr, y_tr, params.xgb_params, params.lgbm_params, n_splits=cfg.cv_splits)

        # Fit final models on the full training window
        xgb = XGBClassifier(**params.xgb_params)
        lgb = LGBMClassifier(**params.lgbm_params)

        X_tr_lgb = X_tr.copy()
        for c in cat_cols:
            X_tr_lgb[c] = X_tr_lgb[c].astype('category')

        xgb.fit(X_tr, y_tr, verbose=False)
        lgb.fit(X_tr_lgb, y_tr)  # <-- Remove verbose=False here

        # Predict on the current point i (today's features predict tomorrow's direction)
        row = df_feat.iloc[[i]]
        X_row = row[feature_cols]
        X_row_lgb = X_row.copy()
        for c in cat_cols:
            X_row_lgb[c] = X_row_lgb[c].astype('category')

        p_xgb = xgb.predict_proba(X_row)[:, 1][0]
        p_lgb = lgb.predict_proba(X_row_lgb)[:, 1][0]
        p_ens = w_xgb * p_xgb + w_lgb * p_lgb

        results.append({
            'Date': dates[i],
            'pred_xgb': p_xgb,
            'pred_lgb': p_lgb,
            'pred_ens': p_ens,
            'w_xgb': w_xgb,
            'w_lgb': w_lgb,
            'y': int(df_feat['y'].iloc[i]),
            'Close': float(df_feat['Close'].iloc[i])
        })

    out = pd.DataFrame(results).set_index('Date')
    return out


# ---------------------------------------------------------------------------
# Metrics & Simple Strategy Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(pred_df: pd.DataFrame, threshold: float = 0.5) -> Dict:
    y_true = pred_df['y'].values
    y_prob = pred_df['pred_ens'].values
    y_hat = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_hat)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_hat, average='binary', zero_division=0)

    # Optional: naive PnL assuming you go long if prob>=thr else flat, next‑day close‑to‑close returns
    # Shift true next‑day direction into realized return
    realized_ret = np.where(y_true==1, 1, -1)  # +1 if up, -1 if down (for illustration only)
    strat_ret = realized_ret * (y_hat*2 - 1)   # +1 if we predicted up, -1 if predicted down
    cum_score = np.cumsum(strat_ret)

    return {
        'accuracy': acc,
        'auc': auc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'cumulative_signed_hits': int(cum_score[-1]) if len(cum_score)>0 else 0
    }


# ---------------------------------------------------------------------------
# Dynamic Threshold
# ---------------------------------------------------------------------------

def optimize_threshold_online(pred_df, price_df, grid=None, lookback=252):
    """
    For each day t, choose the threshold (from `grid`) that would have maximized
    open->close equity over the prior `lookback` days, using past realized returns only.
    Returns a pd.Series of thresholds aligned with pred_df.index.
    """
    if grid is None:
        grid = np.round(np.linspace(0.50, 0.70, 21), 3)

    df = pred_df[['pred_ens']].copy()
    df = df.merge(price_df[['Open','Close']], left_index=True, right_index=True, how='left')
    oc = (df['Close'] / df['Open'] - 1)  # open->close realized return

    thr_series = pd.Series(index=df.index, dtype=float)

    # Precompute cumulative sums for speed
    probs = df['pred_ens'].values
    oc_vals = oc.values
    n = len(df)

    for i in range(n):
        start = max(0, i - lookback)
        if i - start < 20:  # need a minimum history
            thr_series.iloc[i] = 0.60  # conservative default
            continue

        best_thr, best_equity = 0.5, -1e9
        past_probs = probs[start:i]
        past_oc    = oc_vals[start:i]

        for thr in grid:
            sig = (past_probs >= thr).astype(int)
            eq  = np.prod(1 + sig * past_oc)  # equity over lookback
            if eq > best_equity:
                best_equity, best_thr = eq, thr

        thr_series.iloc[i] = best_thr

    return thr_series


# ---------------------------------------------------------------------------
# High Confidence Signal Trading
# ---------------------------------------------------------------------------

def high_confidence_signal(pred_df, q=0.90, lookback=63):
    """
    Rolling percentile filter: signal=1 when today's prob is in the top q
    of the last `lookback` days (uses only past data).
    """
    p = pred_df['pred_ens']
    roll = p.rolling(lookback, min_periods=20)
    # rolling quantile *of past*, so shift(1) to avoid including today
    qpast = roll.quantile(q).shift(1)
    sig = (p >= qpast).astype(int).fillna(0)
    return sig


# ---------------------------------------------------------------------------
# Default hyperparameters (conservative, stable)
# ---------------------------------------------------------------------------

def default_params() -> ModelParams:
    xgb_params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='auc',
        random_state=42,         # set to int for consistency
        n_jobs=-1,
    )

    lgbm_params = dict(
        n_estimators=600,           
        learning_rate=0.03,       
        num_leaves=63,           # leaf‑wise growth (LightGBM strength) 
        max_depth=-1,            # unlimited; we control with num_leaves + min_data_in_leaf 
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        random_state=42,         # set to int for consistency
        n_jobs=-1,
    )

    return ModelParams(xgb_params=xgb_params, lgbm_params=lgbm_params)


# ---------------------------------------------------------------------------
# Equity Curves
# ---------------------------------------------------------------------------

def compute_equity_curves(pred_df: pd.DataFrame,
                          price_df: pd.DataFrame,
                          threshold: float = 0.5,
                          rolling_thresholds: Optional[pd.Series] = None,
                          precomputed_signal: Optional[pd.Series] = None) -> pd.DataFrame:
    df = pred_df.copy()
    df = df.merge(price_df[['Open','Close']], left_index=True, right_index=True, how='left')

    # After merge, handle possible column names for 'Close'
    close_col = None
    if 'Close' in df.columns:
        close_col = 'Close'
    elif 'Close_y' in df.columns:
        close_col = 'Close_y'
    elif 'Close_x' in df.columns:
        close_col = 'Close_x'
    else:
        raise KeyError(f"No 'Close' column found after merge. Columns: {df.columns}")

    c2c = df[close_col].pct_change().fillna(0)
    o2c = (df[close_col] / df['Open'] - 1).fillna(0)

    if precomputed_signal is not None:
        sig = precomputed_signal.reindex(df.index).fillna(0).astype(int)
    elif rolling_thresholds is not None:
        thr = rolling_thresholds.reindex(df.index).fillna(threshold)
        sig = (df['pred_ens'] >= thr).astype(int)
    else:
        sig = (df['pred_ens'] >= threshold).astype(int)

    # Trade today's open->close using **yesterday's** decision
    sig = sig.shift(1).fillna(0).astype(int)

    strat = (1 + sig * o2c).cumprod()
    always = (1 + o2c).cumprod()
    buyhold = (1 + c2c).cumprod()
    return pd.DataFrame({
        'Strategy': strat,
        'AlwaysLong_Intraday': always,
        'BuyHold_Close2Close': buyhold
    }, index=df.index)

    return curves


def plot_equity_curves(curves: pd.DataFrame, ticker: str = ''):
    plt.figure(figsize=(12,6))
    for col in curves.columns:
        plt.plot(curves.index, curves[col], label=col)
    plt.legend()
    plt.title(f"Cumulative Equity Curves {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.grid(True)
    plt.show()

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(df: Optional[pd.DataFrame] = None,
                   ticker: str = 'AAPL',
                   start: str = '2010-01-01',
                   end: Optional[str] = None,
                   cfg: Optional[WalkForwardConfig] = None,
                   params: Optional[ModelParams] = None,
                   threshold: float = 0.5,
                   seed: int = 42,
                   lr_xgb: float = None,
                   lr_lgb: float = None,
                   skip_training: bool = False,
                   plot: bool = True,
                   XGBoost_max_depth_exp: int = 4,
                   LGBM_num_leaves_exp: int = 63) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:

    if params is None:
        params = default_params()
    if lr_xgb is not None:
        params.xgb_params['learning_rate'] = lr_xgb
    if lr_lgb is not None:
        params.lgbm_params['learning_rate'] = lr_lgb
    if XGBoost_max_depth_exp is not None:
        params.xgb_params['max_depth'] = XGBoost_max_depth_exp
    if LGBM_num_leaves_exp is not None:
        params.lgbm_params['num_leaves'] = LGBM_num_leaves_exp

    params.xgb_params['random_state'] = seed
    params.lgbm_params['random_state'] = seed

    if df is None:
        if yf is None:
            raise RuntimeError("yfinance not available; pass a DataFrame instead.")
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data = data[['Open','High','Low','Close','Volume']]
    else:
        data = df.copy()

    feat = make_features(data)

    if skip_training:
        # Fake pred_df: just hold close prices and random probs
        pred_df = pd.DataFrame({
            'pred_ens': np.random.rand(len(feat)),
            'y': feat['y'].values
        }, index=feat.index)
        metrics = {}
    else:
        if params is None:
            params = default_params()
        if cfg is None:
            cfg = WalkForwardConfig()
        pred_df = walk_forward_ensemble(feat, cfg, params)
        metrics = evaluate_predictions(pred_df, threshold=threshold)

    curves = compute_equity_curves(pred_df, data, threshold=threshold)
    
    if plot:
        plot_equity_curves(curves, ticker)

    return pred_df, metrics, curves, data



def sweep_experiments(ticker: str,
                      seeds: List[int] = [42],
                      lrs: List[float] = [0.03],
                      thresholds: List[float] = [0.5],
                      start: str = '2015-01-01',
                      end: Optional[str] = None,
                      cfg: Optional[WalkForwardConfig] = None,
                      are_skip_training: bool = False,
                      XGBoost_max_depth: List[int] = [42],
                      LGBM_num_leaves: List[int] = [42],
                      ):
    """
    Run multiple experiments across seeds, learning rates, and thresholds.
    Plot equity curves (strategy + baselines) for comparison.
    """
    data_cache = {}     # Prevent re-downloading data
    all_curves = {}

    for seed in seeds:
        for lr in lrs:
            for thr in thresholds:
                for XGBoost_max_depth_exp in XGBoost_max_depth:
                    for LGBM_num_leaves_exp in LGBM_num_leaves:
                        label = f"seed={seed},lr={lr},thr={thr},XGBoost_max_depth={XGBoost_max_depth_exp},LGBM_num_leaves={LGBM_num_leaves_exp}"
                        print(f"Running {label} ...")
                        if ticker not in data_cache:
                            preds, metrics, curves, raw_data = run_experiment(
                                ticker=ticker,
                                start=start,
                                end=end,
                                cfg=cfg,
                            seed=seed,
                            lr_xgb=lr,
                            lr_lgb=lr,
                            threshold=thr,
                            plot=False,   # suppress per-run plotting
                            XGBoost_max_depth_exp=XGBoost_max_depth_exp,
                            LGBM_num_leaves_exp=LGBM_num_leaves_exp,
                            skip_training=are_skip_training
                        )
                            data_cache[ticker] = raw_data
                        else:
                            preds, metrics, curves, _ = run_experiment(
                                ticker=ticker,
                                start=start,
                                end=end,
                                cfg=cfg,
                            seed=seed,
                            lr_xgb=lr,
                            lr_lgb=lr,
                            threshold=thr,
                            plot=False,   # suppress per-run plotting
                            XGBoost_max_depth_exp=XGBoost_max_depth_exp,
                            LGBM_num_leaves_exp=LGBM_num_leaves_exp,
                            skip_training=are_skip_training
                        )
                        # Baselines (store once)
                        if 'BuyHold_Close2Close' not in all_curves:
                            all_curves['BuyHold_Close2Close'] = curves['BuyHold_Close2Close']
                        if 'AlwaysLong_Intraday' not in all_curves:
                            all_curves['AlwaysLong_Intraday'] = curves['AlwaysLong_Intraday']

                        # 1) Fixed threshold (as before)
                        all_curves[f"{label} (fixed)"] = curves['Strategy']

                        # 2) Online optimized threshold
                        thr_series = optimize_threshold_online(preds, data_cache[ticker])  # ensure you pass the same raw OHLC df used in run_experiment
                        curves_opt = compute_equity_curves(preds, data_cache[ticker], rolling_thresholds=thr_series)
                        all_curves[f"{label} (online-thr)"] = curves_opt['Strategy']

                        # 3) High-confidence filter (top decile)
                        sig_q = high_confidence_signal(preds, q=0.90, lookback=63)
                        curves_q = compute_equity_curves(preds, data_cache[ticker], precomputed_signal=sig_q)
                        all_curves[f"{label} (top10%)"] = curves_q['Strategy']

    # Plot everything
    plt.figure(figsize=(12,6))
    for label, curve in all_curves.items():
        plt.plot(curve.index, curve.values, label=label)
    plt.legend()
    plt.title(f"Equity Curves Comparison - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.grid(True)
    plt.show()

    return all_curves



# ---------------------------------------------------------------------------
# Script entry point (demo)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid XGBoost + LightGBM walk‑forward stock classifier")
    parser.add_argument('--ticker', type=str, default='GM')
    parser.add_argument('--start', type=str, default='2010-01-01')      # was 2010-01-01
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--start_train_days', type=int, default=504)
    parser.add_argument('--refit_every_n_days', type=int, default=5)
    parser.add_argument('--rolling_window_days', type=int, default=None)
    parser.add_argument('--cv_splits', type=int, default=3)
    args = parser.parse_args()

    cfg = WalkForwardConfig(
        start_train_days=args.start_train_days,
        refit_every_n_days=args.refit_every_n_days,
        rolling_window_days=args.rolling_window_days,
        cv_splits=args.cv_splits,
    )

    '''
    preds, metrics, curves = run_experiment(            # Single run
       ticker=args.ticker,
        start=args.start,
        end=args.end,
        cfg=cfg,
        skip_training=False,
    )

    print("=== Metrics (Ensemble) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nHead of predictions:")
    print(preds.head())
    print("\nTail of predictions:")
    print(preds.tail())
    '''
    curves = sweep_experiments(                         # Testing sweep across params
    ticker="AAPL",
    seeds=[1],
    lrs=[0.03],
    thresholds=[0.5],
    are_skip_training=False,
    XGBoost_max_depth=[4],                               # Originally 4
    LGBM_num_leaves=[63],                                # Originally 63

    )