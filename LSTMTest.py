import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
TICKER = "MSFT"
START = "2015-01-01"
END   = "2023-01-01"
SEQ_LEN = 30
TRAIN_YEARS = 3          # years in the initial training window
STEP_DAYS = 252          # how far to roll forward each fold (~1y)
EPOCHS = 5
LR = 1e-3
THRESHOLD = 0.5          # probability cutoff for "up"
COST_BPS = 0             # set e.g. to 5 for 5 bps per entry + 5 bps per exit (intraday)

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# Data & Labels
# ----------------------------
data = yf.download(TICKER, start=START, end=END)
data = data.dropna().copy()

# Label for day t: 1 if Close_t > Open_t, else 0
data["y"] = (data["Close"] > data["Open"]).astype(int)
# Intraday return for day t (what we actually realize if we go long at open_t and exit at close_t)
data["intraday_ret"] = (data["Close"] - data["Open"]) / data["Open"]

feat_cols = ["Open", "High", "Low", "Close", "Volume"]
raw_features = data[feat_cols].values
labels = data["y"].values
dates = data.index

# ----------------------------
# Helpers
# ----------------------------
def create_sequences(features_2d, labels_1d, seq_len):
    """
    Given contiguous rows for a block (train or test),
    returns sequences of len `seq_len` to predict the NEXT day's label.
    X[i] uses rows [i .. i+seq_len-1]; y[i] is label at i+seq_len.
    """
    X, y = [], []
    n = len(features_2d)
    for i in range(n - seq_len):
        X.append(features_2d[i:i+seq_len])
        y.append(labels_1d[i+seq_len])
    return np.array(X), np.array(y)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ----------------------------
# Walk-forward (expanding window)
# ----------------------------
def walk_forward_validation(raw_features, labels, dates, seq_len=30, train_years=3, step_days=252,
                            epochs=5, lr=1e-3, threshold=0.5):
    n = len(raw_features)
    train_min = train_years * 252
    results = []

    # folds iterate by the end-of-train index (exclusive), expanding from the beginning
    for train_end in range(train_min, n - seq_len, step_days):
        test_end = min(train_end + step_days, n)  # exclusive

        # Raw blocks
        X_train_raw = raw_features[:train_end]
        y_train_raw = labels[:train_end]
        X_test_raw  = raw_features[train_end:test_end]
        y_test_raw  = labels[train_end:test_end]

        # Scale *only* on training, transform both sets
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled  = scaler.transform(X_test_raw)

        # Build sequences
        Xtr, ytr = create_sequences(X_train_scaled, y_train_raw, seq_len)
        Xte, yte = create_sequences(X_test_scaled,  y_test_raw,  seq_len)
        if len(Xte) == 0:
            break  # not enough test samples for a seq

        # To tensors
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        yte_t = torch.tensor(yte, dtype=torch.float32).unsqueeze(-1)

        # Train a fresh model each fold
        model = LSTMClassifier()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            out = model(Xtr_t)
            loss = criterion(out, ytr_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            proba = model(Xte_t).numpy().flatten()
        pred = (proba >= threshold).astype(int)

        # Map predictions to absolute calendar dates
        # First prediction corresponds to absolute index: train_end + seq_len
        start_idx = train_end + seq_len
        pred_dates = dates[start_idx : start_idx + len(pred)]

        # Fold metrics
        acc = accuracy_score(yte, pred)
        print(f"Fold: {dates[0].date()}–{dates[train_end-1].date()} → "
              f"test {dates[train_end].date()}–{dates[min(test_end-1, n-1)].date()} | "
              f"n_test={len(pred)} | acc={acc:.3f}")

        # Save fold results
        results.append(pd.DataFrame({
            "Pred": pred,
            "Proba": proba,
            "True": yte,
        }, index=pred_dates))

    return pd.concat(results).sort_index()

# ----------------------------
# Run WF and Backtest
# ----------------------------
wf = walk_forward_validation(
    raw_features, labels, dates,
    seq_len=SEQ_LEN,
    train_years=TRAIN_YEARS,
    step_days=STEP_DAYS,
    epochs=EPOCHS,
    lr=LR,
    threshold=THRESHOLD
)

# Strategy: go long intraday (open→close) on days where Pred == 1
intraday = data["intraday_ret"].reindex(wf.index)
signal = wf["Pred"].astype(int)

# Optional costs: intraday trade means you enter + exit on each traded day → 2 legs
# COST_BPS = 5 means cost fraction per leg = 0.0005, total per traded day = 0.001
if COST_BPS and COST_BPS > 0:
    cost_frac_per_day = 2 * (COST_BPS / 1e4)
else:
    cost_frac_per_day = 0.0

strategy_ret = intraday * signal - cost_frac_per_day * signal
equity = (1 + strategy_ret.fillna(0)).cumprod()

# Baselines
intraday_always = (1 + intraday.fillna(0)).cumprod()              # always long intraday
buy_hold_close = (1 + data["Close"].pct_change().reindex(wf.index).fillna(0)).cumprod()  # close→close

# ----------------------------
# Plotting
# ----------------------------

plt.figure(figsize=(14,6))
plt.plot(equity, label="Strategy (intraday)")
plt.plot(intraday_always, label="Always long (intraday)")
plt.plot(buy_hold_close, label="Buy & Hold (close→close)")

# Highlight days where strategy is flat (Pred = 0)
flat_indices = equity.index[signal == 0]
for day in flat_indices:
    plt.axvspan(day, day, color='red', alpha=0.15)

plt.title(f"{TICKER} – Strategy vs Always-long: Flat days highlighted")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Diagnostics
# ----------------------------
def summarize_performance(strategy_ret, intraday_ret, buyhold_ret, signal):
    summary = {}

    # How often do we trade?
    trades = signal.sum()
    total_days = len(signal)
    summary["% days traded"] = trades / total_days * 100

    # Hit ratio = correct direction on days we traded
    mask = signal == 1
    correct = (intraday_ret[mask] > 0).sum()
    summary["Hit ratio"] = correct / trades * 100 if trades > 0 else np.nan

    # Average returns
    summary["Avg daily return (strategy)"] = strategy_ret.mean()
    summary["Avg daily return (always intraday)"] = intraday_ret.mean()
    summary["Avg daily return (buy&hold close-close)"] = buyhold_ret.mean()

    # Sharpe ratio (force Series)
    def sharpe(r):
        r = pd.Series(r).dropna()
        std = r.std()
        if std == 0 or np.isnan(std):
            return np.nan
        return r.mean() / std * np.sqrt(252)

    summary["Sharpe (strategy)"] = sharpe(strategy_ret)
    summary["Sharpe (always intraday)"] = sharpe(intraday_ret)
    summary["Sharpe (buy&hold)"] = sharpe(buyhold_ret)

    # CAGR (compound annual growth rate)
    def CAGR(equity):
        equity = pd.Series(equity).dropna()
        years = len(equity) / 252
        return equity.iloc[-1]**(1/years) - 1 if len(equity) > 0 else np.nan

    summary["CAGR (strategy)"] = CAGR((1+strategy_ret).cumprod())
    summary["CAGR (always intraday)"] = CAGR((1+intraday_ret).cumprod())
    summary["CAGR (buy&hold)"] = CAGR((1+buyhold_ret).cumprod())

    return pd.Series(summary)

# Align returns
buyhold_ret = data["Close"].pct_change().reindex(wf.index)
summary = summarize_performance(
    strategy_ret.fillna(0).squeeze(),
    intraday.fillna(0).squeeze(),
    buyhold_ret.fillna(0).squeeze(),
    signal
)

print("\nPerformance Summary:")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))