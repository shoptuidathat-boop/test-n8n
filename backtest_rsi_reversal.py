#!/usr/bin/env python3
"""
RSI Reversal Strategy Backtest
Pair: DOGE/USDT | Timeframe: 15m
"""

import os
import time
import json
import ccxt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# ── CONFIG ───────────────────────────────────────────────────────────────
SYMBOL = "DOGE/USDT"
TIMEFRAME = "15m"
RSI_LENGTH = 14
SL_PCT = 0.0015       # 0.15%
TP_PCT = 0.003        # 0.30%
INITIAL_CAPITAL = 10_000.0
YEARS_OF_DATA = 3     # fetch ~3 years

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "doge_usdt_15m.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── DATA FETCHING ────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    """Fetch historical OHLCV via ccxt (OKX), paginating as needed."""
    if os.path.exists(DATA_FILE):
        print(f"Loading cached data from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        print(f"  Loaded {len(df)} candles  ({df['timestamp'].min()} → {df['timestamp'].max()})")
        return df

    exchange = ccxt.okx({"enableRateLimit": True})
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - years * 365.25 * 24 * 3600 * 1000
    since = int(since)
    limit = 100  # OKX returns max 100 per request
    all_rows = []

    print(f"Fetching {symbol} {timeframe} from OKX …")
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        since = last_ts + ms_per_candle
        if since >= exchange.milliseconds():
            break
        if len(ohlcv) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)
        if len(all_rows) % 10000 == 0:
            print(f"  … {len(all_rows)} candles fetched so far")

    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df.drop(columns=["timestamp_ms"], inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"  Saved {len(df)} candles to {DATA_FILE}")
    return df


# ── INDICATORS ───────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-smoothed RSI (matches TradingView default)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ── BACKTEST ENGINE ──────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], RSI_LENGTH)

    trades = []
    equity = [INITIAL_CAPITAL]
    capital = INITIAL_CAPITAL

    i = 2  # need at least index 0=prev-prev, 1=prev, 2=current for the setup check
    while i < len(df) - 1:
        rsi_prev = df.at[i - 1, "rsi"]
        rsi_curr = df.at[i, "rsi"]
        high_prev = df.at[i - 1, "high"]
        high_curr = df.at[i, "high"]
        low_prev = df.at[i - 1, "low"]
        low_curr = df.at[i, "low"]

        direction = None

        # ── LONG CONDITION ───────────────────────────────────────────
        if rsi_prev <= 20 and rsi_curr > rsi_prev and low_curr >= low_prev:
            direction = "LONG"

        # ── SHORT CONDITION ──────────────────────────────────────────
        elif rsi_prev >= 80 and rsi_curr < rsi_prev and high_curr <= high_prev:
            direction = "SHORT"

        if direction is None:
            equity.append(capital)
            i += 1
            continue

        # ── CONFIRMATION CANDLE (next candle after setup) ────────────
        confirm_idx = i + 1
        if confirm_idx >= len(df):
            equity.append(capital)
            i += 1
            continue

        confirm = df.iloc[confirm_idx]

        # Cancel checks during confirmation candle
        if direction == "LONG":
            if confirm["low"] < low_prev:
                equity.append(capital)
                i = confirm_idx + 1
                continue
            if confirm["close"] <= confirm["open"]:
                equity.append(capital)
                i = confirm_idx + 1
                continue

        if direction == "SHORT":
            if confirm["high"] > high_prev:
                equity.append(capital)
                i = confirm_idx + 1
                continue
            if confirm["close"] >= confirm["open"]:
                equity.append(capital)
                i = confirm_idx + 1
                continue

        # ── ENTRY ────────────────────────────────────────────────────
        entry_price = confirm["close"]

        if direction == "LONG":
            sl = entry_price * (1 - SL_PCT)
            tp = entry_price * (1 + TP_PCT)
        else:
            sl = entry_price * (1 + SL_PCT)
            tp = entry_price * (1 - TP_PCT)

        # ── SIMULATE TRADE bar-by-bar after entry ────────────────────
        exit_price = None
        exit_reason = None
        exit_idx = None

        for j in range(confirm_idx + 1, len(df)):
            bar = df.iloc[j]
            if direction == "LONG":
                if bar["low"] <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    exit_idx = j
                    break
                if bar["high"] >= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    exit_idx = j
                    break
            else:  # SHORT
                if bar["high"] >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                    exit_idx = j
                    break
                if bar["low"] <= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    exit_idx = j
                    break

        if exit_price is None:
            # Trade never hit SL/TP → close at last bar
            exit_price = df.iloc[-1]["close"]
            exit_reason = "END"
            exit_idx = len(df) - 1

        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        pnl = capital * pnl_pct
        capital += pnl

        trades.append({
            "direction": direction,
            "entry_time": str(df.at[confirm_idx, "timestamp"]),
            "entry_price": round(entry_price, 8),
            "exit_time": str(df.at[exit_idx, "timestamp"]),
            "exit_price": round(exit_price, 8),
            "exit_reason": exit_reason,
            "pnl_pct": round(pnl_pct * 100, 4),
            "pnl": round(pnl, 4),
            "capital_after": round(capital, 4),
        })

        equity.append(capital)
        i = exit_idx + 1
        continue

    # Pad equity to match df length for alignment
    while len(equity) < len(df):
        equity.append(capital)

    return {"trades": trades, "equity": equity[:len(df)], "final_capital": capital}


# ── METRICS ──────────────────────────────────────────────────────────────
def compute_metrics(result: dict) -> dict:
    trades = result["trades"]
    equity = np.array(result["equity"])

    if not trades:
        return {"total_trades": 0}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    # Long / Short breakdown
    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "winrate_pct": round(len(wins) / len(trades) * 100, 2),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else float("inf"),
        "max_drawdown_pct": round(max_dd, 4),
        "net_profit": round(result["final_capital"] - INITIAL_CAPITAL, 2),
        "net_profit_pct": round((result["final_capital"] / INITIAL_CAPITAL - 1) * 100, 2),
        "final_capital": round(result["final_capital"], 2),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_wins": len([t for t in longs if t["pnl"] > 0]),
        "short_wins": len([t for t in shorts if t["pnl"] > 0]),
        "avg_pnl_pct": round(np.mean([t["pnl_pct"] for t in trades]), 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


# ── PLOTTING ─────────────────────────────────────────────────────────────
def plot_equity_curve(df: pd.DataFrame, equity: list, metrics: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    timestamps = df["timestamp"].values[:len(equity)]
    ax.plot(timestamps, equity, linewidth=0.8, color="#2196F3")
    ax.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(
        f"RSI Reversal – {SYMBOL} 15m | "
        f"Trades: {metrics['total_trades']}  WR: {metrics['winrate_pct']}%  "
        f"PF: {metrics['profit_factor']}  MDD: {metrics['max_drawdown_pct']:.2f}%",
        fontsize=11,
    )
    ax.set_ylabel("Equity (USDT)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Equity curve saved → {out_path}")


# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    df = fetch_ohlcv(SYMBOL, TIMEFRAME, YEARS_OF_DATA)

    # Validate no missing candles
    expected_gap = pd.Timedelta(minutes=15)
    gaps = df["timestamp"].diff().dropna()
    bad = gaps[gaps > expected_gap * 1.5]
    if not bad.empty:
        print(f"  WARNING: {len(bad)} gaps > 22.5 min detected (largest: {bad.max()})")
    else:
        print("  No missing candles detected.")

    result = run_backtest(df)
    metrics = compute_metrics(result)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save trades
    trades_path = os.path.join(RESULTS_DIR, "trades.csv")
    pd.DataFrame(result["trades"]).to_csv(trades_path, index=False)

    # Equity curve chart
    chart_path = os.path.join(RESULTS_DIR, "equity_curve.png")
    plot_equity_curve(df, result["equity"], metrics, chart_path)

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS – RSI Reversal Strategy")
    print(f"Pair: {SYMBOL}  |  Timeframe: {TIMEFRAME}")
    print(f"Period: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Candles: {len(df)}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
