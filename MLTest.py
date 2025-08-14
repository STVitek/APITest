# pip install yfinance pandas scikit-learn matplotlib

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ---------- helpers ----------
def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ema(series: pd.Series, span: int) -> pd.Series:
    series = series.astype(float)
    return series.ewm(span=span, adjust=False).mean()

# ---------- params ----------
ticker = "AAPL"
start_date = "2024-01-01"
end_date   = "2024-12-31"

# ---------- data ----------
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
df = df.sort_index()

# Ensure Close is a 1-D numeric Series
if isinstance(df["Close"], pd.DataFrame):
    close = df["Close"].iloc[:, 0].astype(float)
else:
    close = df["Close"].astype(float)

df['Close'] = close  # keep Close aligned with df

# ---------- features ----------
df["Return"] = df['Close'].pct_change()
df["RSI"]    = rsi_wilder(df['Close'], window=14)
df["EMA_10"] = ema(df['Close'], span=10)
df["EMA_30"] = ema(df['Close'], span=30)
df["Target"] = (df['Close'].shift(-1) > df['Close']).astype(int)

# clean
df = df.dropna()

# ---------- model ----------
features = ["Return", "RSI", "EMA_10", "EMA_30"]
X = df[features]
y = df["Target"]

split = int(len(df) * 0.50)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:\n", fi)

# ---------- add predictions ----------
df['Pred'] = 0
df.iloc[split:, df.columns.get_loc('Pred')] = y_pred
df['Correct'] = df['Target'] == df['Pred']

# ---------- visualization ----------
plt.figure(figsize=(16,6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')

# Function to group consecutive correct/incorrect predictions
def highlight_streaks(df, start_idx, color_correct='green', color_incorrect='red', alpha_correct=0.2, alpha_incorrect=0.1):
    current_state = None
    streak_start = start_idx
    
    for i in range(start_idx, len(df)):
        state = df['Correct'].iloc[i]
        
        if current_state is None:
            current_state = state
            streak_start = i
        elif state != current_state:
            # Draw previous streak
            if current_state:  # correct
                plt.axvspan(df.index[streak_start], df.index[i-1], color=color_correct, alpha=alpha_correct)
            else:  # incorrect
                plt.axvspan(df.index[streak_start], df.index[i-1], color=color_incorrect, alpha=alpha_incorrect)
            # Start new streak
            streak_start = i
            current_state = state
    
    # Draw last streak
    if current_state:
        plt.axvspan(df.index[streak_start], df.index[len(df)-1], color=color_correct, alpha=alpha_correct)
    else:
        plt.axvspan(df.index[streak_start], df.index[len(df)-1], color=color_incorrect, alpha=alpha_incorrect)

# Apply highlighting from test set onward
highlight_streaks(df, start_idx=split)

# Optional: plot EMA lines for trend context
plt.plot(df.index, df['EMA_10'], label='EMA 10', color='cyan', alpha=0.7)
plt.plot(df.index, df['EMA_30'], label='EMA 30', color='magenta', alpha=0.7)

plt.title(f"{ticker} Close Price with Correct Prediction Highlights (Grouped Streaks)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
