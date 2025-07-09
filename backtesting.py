import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# Load your trained model
with open("xgboost_top5_btc_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load historical data
df = pd.read_csv("Bitcoin_BTCUSDT.csv").head(-100) # Your minute-level data

df['mid_price'] = (df['high'] + df['low']) / 2
df['spread'] = df['high'] - df['low']
df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
df['rsi_14'] = df['close'].rolling(window=14).apply(lambda x: (100 - (100 / (1 + (x.mean() / x.std())))) if x.std() != 0 else 0)
# Initialize
initial_balance = 10000  # USDT
inventory = 0  # Base asset
balance = initial_balance
maker_fee = 0.001  # 0.1%
spread_scale = 0.5  # Use to adjust bid/ask width

# Store trade log
trades = []

for i in range(1, len(df)):
    row = df.iloc[i]
    
    # Prepare input for model
    features = row[["mid_price", "ema_15", "ema_5", "rsi_14", "spread"]].values.reshape(1, -1)
    predicted_change = model.predict(features)[0]
    
    # Adjust skew: positive prediction = raise ask, negative = lower bid
    skew = spread_scale * predicted_change
    
    # Fair value
    mid_price = row["mid_price"]

    # Set bid and ask prices
    bid_price = mid_price - (row["spread"] / 2) + skew
    ask_price = mid_price + (row["spread"] / 2) + skew

    # Simulate fill logic (simple: assume price crossed = filled)
    next_close = df.iloc[i + 1]["close"] if i + 1 < len(df) else row["close"]
    
    filled_buy = next_close <= bid_price
    filled_sell = next_close >= ask_price

    # Execute trade
    if filled_buy:
        amount = 1
        cost = bid_price * amount
        if balance >= cost:
            inventory += amount
            balance -= cost * (1 + maker_fee)
            trades.append({"side": "buy", "price": bid_price, "amount": amount, "timestamp": row["timestamp"]})
    
    if filled_sell and inventory > 0:
        amount = 1
        revenue = ask_price * amount
        inventory -= amount
        balance += revenue * (1 - maker_fee)
        trades.append({"side": "sell", "price": ask_price, "amount": amount, "timestamp": row["timestamp"]})

# Final PnL
final_value = balance + inventory * df.iloc[-1]["close"]
returns = (final_value - initial_balance) / initial_balance

print(f"Final Portfolio Value: {final_value:.2f} USDT")
print(f"Return: {returns*100:.2f}%")

# Optional: save trades to CSV
pd.DataFrame(trades).to_csv("backtest_trades.csv", index=False)
