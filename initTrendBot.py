import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------
# Feature engineering
# --------------------------
def make_features(df):
    df = df.copy()
    df["ret_oc"] = (df["Close"] / df["Open"]) - 1
    df["ret_cc"] = df["Close"].pct_change()
    df["ret_oo"] = df["Open"].pct_change()

    for win in [5, 10, 20, 50]:
        df[f"ma_{win}"] = df["Close"].rolling(win).mean()
        df[f"ma_ratio_{win}"] = df["Close"] / df[f"ma_{win}"]
        df[f"mom_{win}"] = df["Close"].pct_change(win)

    df["volatility_20"] = df["ret_cc"].rolling(20).std()
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (
        df["Volume"].rolling(20).std()
    )
    return df


# --------------------------
# Labeling
# --------------------------
def make_labels(df):
    df = df.copy()
    df["y"] = (df["Close"].shift(-1) > df["Open"].shift(-1)).astype(int)
    return df


# --------------------------
# Walk-forward split
# --------------------------
def year_walk_forward(df, min_train=2, n_splits=6):
    years = sorted(df.index.year.unique())
    max_splits = len(years) - min_train
    actual_splits = min(n_splits, max_splits)

    for i in range(min_train, min_train + actual_splits):
        train_years = years[:i]
        test_year = years[i]
        train_idx = df[df.index.year.isin(train_years)].index
        test_idx = df[df.index.year == test_year].index
        yield train_idx, test_idx


# --------------------------
# Strategy backtest
# --------------------------
def backtest_signals(df, prob, prob_long=0.55, prob_flat=0.50, cost_bps=5):
    signals = np.where(prob >= prob_long, 1, 0)
    signals[(prob < prob_flat)] = 0

    df = df.copy()
    df["signal"] = signals
    df["fwd_ret"] = df["Close"].shift(-1) / df["Open"].shift(-1) - 1
    df["strat_ret"] = df["signal"] * df["fwd_ret"]

    # transaction cost per trade
    cost = cost_bps / 10000
    df["trade"] = df["signal"].diff().fillna(0).abs()
    df["strat_ret"] -= df["trade"] * cost
    return df


# --------------------------
# Metrics
# --------------------------
def perf_metrics(returns):
    cumret = (1 + returns).prod() - 1
    ann_ret = (1 + cumret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = (returns.add(1).cumprod() / (returns.add(1).cumprod().cummax()) - 1).min()
    return {"AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "MaxDD": dd}


# --------------------------
# Main experiment runner
# --------------------------
def run_experiment(
    df, cost_bps=5, prob_long=0.55, prob_flat=0.50, min_train_years=2, n_splits=6
):
    df = make_features(df)
    df = make_labels(df).dropna()

    X = df.drop(columns=["y"])
    y = df["y"]
    feature_cols = [
        c for c in df.columns if c not in ["y", "signal", "fwd_ret", "strat_ret", "trade"]
    ]

    models = {
        "LogReg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
    }

    all_results = []
    equity_curves = {}

    for train_idx, test_idx in year_walk_forward(
        df, min_train=min_train_years, n_splits=n_splits
    ):
        X_train, y_train = X.loc[train_idx, feature_cols], y.loc[train_idx]
        X_test, y_test   = X.loc[test_idx, feature_cols], y.loc[test_idx]


        for name, model in models.items():
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            out = backtest_signals(
                df.loc[test_idx], prob, prob_long, prob_flat, cost_bps
            )
            metrics = perf_metrics(out["strat_ret"].dropna())
            metrics.update({"Model": name, "Year": X_test.index.year[0]})
            all_results.append(metrics)

            if name not in equity_curves:
                equity_curves[name] = []
            equity_curves[name].append(out["strat_ret"])

    results = pd.DataFrame(all_results)

    # Merge yearly returns into full equity curves
    full_curves = {}
    for model, rets in equity_curves.items():
        full_curve = pd.concat(rets).sort_index()
        full_curves[model] = full_curve

    return results, full_curves


# --------------------------
# Plotting
# --------------------------
def plot_equity_curves(curves, title="Equity Curves"):
    plt.figure(figsize=(10, 6))
    for model, rets in curves.items():
        (1 + rets).cumprod().plot(label=model)
    plt.legend()
    plt.title(title)
    plt.ylabel("Cumulative Growth")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curves.png")
    plt.show()


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--cost_bps", type=float, default=5)
    parser.add_argument("--prob_long", type=float, default=0.55)
    parser.add_argument("--prob_flat", type=float, default=0.50)
    parser.add_argument("--min_train_years", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=6)
    args = parser.parse_args()

    # Fetch data
    data = yf.download(args.ticker, start=args.start, end=args.end)

    # flatten MultiIndex (e.g. ("Close","AAPL") -> "Close")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.to_csv(f"{args.ticker}_ohlcv.csv", index_label="Date")

    # Run experiment
    results, curves = run_experiment(
        data,
        cost_bps=args.cost_bps,
        prob_long=args.prob_long,
        prob_flat=args.prob_flat,
        min_train_years=args.min_train_years,
        n_splits=args.n_splits,
    )
    print(results)
    results.to_csv("perf_summary.csv", index=False)

    # Plot equity curves
    plot_equity_curves(curves, title=f"{args.ticker} Strategy Equity Curves")