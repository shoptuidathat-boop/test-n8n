#!/usr/bin/env python3
"""
DOGE/USDT 15m Market Structure Analysis
========================================
Discovers recurring paths of market behavior:
Continuation <-> Redistribution ("state path" / "loi mon")

Steps:
  1. Detect swing highs/lows (fractal)
  2. Label micro states (HH, LH, HL, LL)
  3. Build structural states (CONTINUATION_UP, CONTINUATION_DOWN, REDISTRIBUTION)
  4. Build state path sequence
  5. Path analysis (transitions, patterns, lengths, repetitions)
  6. RSI relational behavior per state

Output: state counts, transition matrix, top recurring paths,
        avg lengths, example segments, RSI patterns.
"""

import time
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 0. DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol="DOGE/USDT", timeframe="15m", months=12):
    """Fetch candle data. Tries multiple exchanges for availability."""
    # Try exchanges in order of preference
    exchange_configs = [
        ("bybit", {}),
        ("okx", {}),
        ("kucoin", {}),
        ("binance", {}),
    ]
    exchange = None
    for name, cfg in exchange_configs:
        try:
            cfg["enableRateLimit"] = True
            ex = getattr(ccxt, name)(cfg)
            ex.load_markets()
            if symbol in ex.markets:
                exchange = ex
                print(f"  Using exchange: {name}")
                break
        except Exception as e:
            print(f"  {name} unavailable: {e}")
            continue
    if exchange is None:
        raise RuntimeError(f"No available exchange found for {symbol}")
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - pd.DateOffset(months=months))
        .isoformat()
    )
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_candles = []
    limit = 300  # safe for all exchanges (OKX max is 300)
    batch = 0
    while since < now_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        new_since = candles[-1][0] + 1
        if new_since <= since:
            break  # no progress
        since = new_since
        batch += 1
        if batch % 50 == 0:
            print(f"    ... fetched {len(all_candles)} candles so far")
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def check_missing_candles(df, interval_ms=15 * 60 * 1000):
    """Report gaps larger than one interval."""
    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > interval_ms * 1.5]
    return gaps


# ---------------------------------------------------------------------------
# 1. RSI (Wilder, period=14)
# ---------------------------------------------------------------------------

def wilder_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# 2. SWING DETECTION (fractal)
# ---------------------------------------------------------------------------

def detect_swings(df):
    """
    Swing Low:  low[i] < low[i-1] AND low[i] < low[i+1]
    Swing High: high[i] > high[i-1] AND high[i] > high[i+1]
    Returns ordered list of dicts sorted by index.
    """
    swings = []
    for i in range(1, len(df) - 1):
        low_prev, low_cur, low_next = df["low"].iloc[i - 1], df["low"].iloc[i], df["low"].iloc[i + 1]
        high_prev, high_cur, high_next = df["high"].iloc[i - 1], df["high"].iloc[i], df["high"].iloc[i + 1]

        if low_cur < low_prev and low_cur < low_next:
            swings.append({
                "index": i,
                "type": "low",
                "price": low_cur,
                "timestamp": df["timestamp"].iloc[i],
                "datetime": df["datetime"].iloc[i],
                "rsi": df["rsi"].iloc[i],
            })
        if high_cur > high_prev and high_cur > high_next:
            swings.append({
                "index": i,
                "type": "high",
                "price": high_cur,
                "timestamp": df["timestamp"].iloc[i],
                "datetime": df["datetime"].iloc[i],
                "rsi": df["rsi"].iloc[i],
            })

    swings.sort(key=lambda s: s["index"])
    return swings


# ---------------------------------------------------------------------------
# 3. STEP 1 — LABEL MICRO STATE PER SWING
# ---------------------------------------------------------------------------

def label_micro_states(swings):
    """Compare each swing to the previous swing of the same type."""
    last_high = None
    last_low = None
    for s in swings:
        if s["type"] == "high":
            if last_high is not None:
                s["label"] = "HH" if s["price"] > last_high["price"] else "LH"
                s["rsi_prev"] = last_high["rsi"]
            else:
                s["label"] = None
                s["rsi_prev"] = None
            last_high = s
        else:  # low
            if last_low is not None:
                s["label"] = "HL" if s["price"] > last_low["price"] else "LL"
                s["rsi_prev"] = last_low["rsi"]
            else:
                s["label"] = None
                s["rsi_prev"] = None
            last_low = s
    return swings


# ---------------------------------------------------------------------------
# 4. STEP 2 & 3 — STRUCTURAL STATE + STATE PATH
# ---------------------------------------------------------------------------

# States
UP = "CONTINUATION_UP"
DOWN = "CONTINUATION_DOWN"
REDIST = "REDISTRIBUTION"


def build_state_path(swings):
    """
    Walk labelled swings and classify structural state.

    CONTINUATION_UP   : HH + HL sequence
    CONTINUATION_DOWN : LH + LL sequence
    REDISTRIBUTION    : structural break confirmed by next swing continuing
                        opposite direction.

    Returns list of state-segments:
      {state, start_idx, end_idx, start_dt, end_dt, swing_count, swings}
    """
    labelled = [s for s in swings if s["label"] is not None]
    if len(labelled) < 2:
        return []

    # Build a running view of the latest high-label and low-label
    current_state = None
    segments = []
    seg_swings = []
    last_high_label = None
    last_low_label = None

    for s in labelled:
        if s["type"] == "high":
            last_high_label = s["label"]
        else:
            last_low_label = s["label"]

        # Determine implied structural state from latest pair
        if last_high_label is None or last_low_label is None:
            implied = None
        elif last_high_label == "HH" and last_low_label == "HL":
            implied = UP
        elif last_high_label == "LH" and last_low_label == "LL":
            implied = DOWN
        else:
            implied = REDIST

        if implied is None:
            seg_swings.append(s)
            continue

        if current_state is None:
            current_state = implied
            seg_swings.append(s)
            continue

        if implied == current_state:
            seg_swings.append(s)
        else:
            # State changed — close previous segment
            if seg_swings:
                segments.append(_make_segment(current_state, seg_swings))
            current_state = implied
            seg_swings = [s]

    # Close last segment
    if seg_swings and current_state is not None:
        segments.append(_make_segment(current_state, seg_swings))

    return segments


def _make_segment(state, swings):
    return {
        "state": state,
        "start_idx": swings[0]["index"],
        "end_idx": swings[-1]["index"],
        "start_dt": str(swings[0]["datetime"]),
        "end_dt": str(swings[-1]["datetime"]),
        "swing_count": len(swings),
        "swings": swings,
    }


# ---------------------------------------------------------------------------
# 5. STEP 4 — PATH ANALYSIS
# ---------------------------------------------------------------------------

def path_analysis(segments):
    results = {}

    states = [seg["state"] for seg in segments]

    # 5a. Transition frequency
    transitions = Counter()
    for i in range(len(states) - 1):
        transitions[(states[i], states[i + 1])] += 1
    results["transitions"] = dict(transitions)

    # 5b. Transition matrix (probabilities)
    from_counts = Counter()
    for (src, _), cnt in transitions.items():
        from_counts[src] += cnt
    prob_matrix = {}
    for (src, dst), cnt in transitions.items():
        prob_matrix[(src, dst)] = round(cnt / from_counts[src], 4)
    results["transition_probabilities"] = prob_matrix

    # 5c. Path patterns (length 3)
    patterns_3 = Counter()
    for i in range(len(states) - 2):
        pat = (states[i], states[i + 1], states[i + 2])
        patterns_3[pat] += 1
    results["path_patterns_3"] = dict(patterns_3.most_common(20))

    # Path patterns (length 4)
    patterns_4 = Counter()
    for i in range(len(states) - 3):
        pat = (states[i], states[i + 1], states[i + 2], states[i + 3])
        patterns_4[pat] += 1
    results["path_patterns_4"] = dict(patterns_4.most_common(10))

    # 5d. Length analysis
    lengths = defaultdict(list)
    for seg in segments:
        lengths[seg["state"]].append(seg["swing_count"])
    length_stats = {}
    for st, vals in lengths.items():
        length_stats[st] = {
            "count": len(vals),
            "avg_swings": round(np.mean(vals), 2),
            "median_swings": round(np.median(vals), 2),
            "min_swings": int(np.min(vals)),
            "max_swings": int(np.max(vals)),
        }
    results["length_stats"] = length_stats

    # 5e. Repetition patterns (specific)
    rep = Counter()
    for i in range(len(states) - 2):
        trip = (states[i], states[i + 1], states[i + 2])
        # DOWN -> REDIST -> DOWN (failed reversal)
        if trip == (DOWN, REDIST, DOWN):
            rep["DOWN->REDIST->DOWN (failed reversal)"] += 1
        # DOWN -> REDIST -> UP (true reversal)
        elif trip == (DOWN, REDIST, UP):
            rep["DOWN->REDIST->UP (true reversal)"] += 1
        # UP -> REDIST -> UP (failed reversal)
        elif trip == (UP, REDIST, UP):
            rep["UP->REDIST->UP (failed reversal)"] += 1
        # UP -> REDIST -> DOWN (true reversal)
        elif trip == (UP, REDIST, DOWN):
            rep["UP->REDIST->DOWN (true reversal)"] += 1
    results["repetition_patterns"] = dict(rep)

    return results


# ---------------------------------------------------------------------------
# 6. STEP 5 — RSI RELATIONAL BEHAVIOR
# ---------------------------------------------------------------------------

def rsi_behavior(segments):
    """
    For each state, record:
      RSI_high_n vs RSI_high_{n-1}
      RSI_low_n  vs RSI_low_{n-1}
    """
    rsi_patterns = defaultdict(lambda: {"high_rising": 0, "high_falling": 0,
                                         "low_rising": 0, "low_falling": 0,
                                         "high_total": 0, "low_total": 0})
    for seg in segments:
        for s in seg["swings"]:
            if s.get("rsi_prev") is None or pd.isna(s["rsi"]) or pd.isna(s["rsi_prev"]):
                continue
            key = seg["state"]
            if s["type"] == "high":
                rsi_patterns[key]["high_total"] += 1
                if s["rsi"] > s["rsi_prev"]:
                    rsi_patterns[key]["high_rising"] += 1
                else:
                    rsi_patterns[key]["high_falling"] += 1
            else:
                rsi_patterns[key]["low_total"] += 1
                if s["rsi"] > s["rsi_prev"]:
                    rsi_patterns[key]["low_rising"] += 1
                else:
                    rsi_patterns[key]["low_falling"] += 1

    # Compute ratios
    result = {}
    for state, data in rsi_patterns.items():
        entry = {}
        if data["high_total"] > 0:
            entry["high_rising_pct"] = round(data["high_rising"] / data["high_total"] * 100, 1)
            entry["high_falling_pct"] = round(data["high_falling"] / data["high_total"] * 100, 1)
            entry["high_sample_size"] = data["high_total"]
        if data["low_total"] > 0:
            entry["low_rising_pct"] = round(data["low_rising"] / data["low_total"] * 100, 1)
            entry["low_falling_pct"] = round(data["low_falling"] / data["low_total"] * 100, 1)
            entry["low_sample_size"] = data["low_total"]
        result[state] = entry
    return result


# ---------------------------------------------------------------------------
# 7. EXAMPLE SEGMENTS
# ---------------------------------------------------------------------------

def pick_examples(segments, n=3):
    """Pick up to n example segments for each state type."""
    by_state = defaultdict(list)
    for seg in segments:
        by_state[seg["state"]].append(seg)
    examples = {}
    for state, segs in by_state.items():
        chosen = segs[:n]
        examples[state] = [
            {
                "start_dt": s["start_dt"],
                "end_dt": s["end_dt"],
                "swing_count": s["swing_count"],
                "swing_labels": [sw["label"] for sw in s["swings"]],
            }
            for s in chosen
        ]
    return examples


# ---------------------------------------------------------------------------
# PRETTY PRINTING
# ---------------------------------------------------------------------------

SHORT = {UP: "UP_CONT", DOWN: "DOWN_CONT", REDIST: "REDIST"}


def short(s):
    return SHORT.get(s, s)


def print_report(segments, pa, rsi_pat, examples, df, gaps):
    print("=" * 72)
    print("  DOGE/USDT 15m — MARKET STRUCTURE PATH ANALYSIS")
    print("=" * 72)
    print()

    # Data summary
    print(f"Data range : {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    print(f"Candles    : {len(df)}")
    print(f"Gaps > 15m : {len(gaps)}")
    print()

    # 1. State counts
    print("-" * 40)
    print("1. STATE COUNTS")
    print("-" * 40)
    stats = pa["length_stats"]
    rows = []
    for state in [UP, DOWN, REDIST]:
        if state in stats:
            s = stats[state]
            rows.append([short(state), s["count"], s["avg_swings"],
                         s["median_swings"], s["min_swings"], s["max_swings"]])
    print(tabulate(rows, headers=["State", "Count", "Avg Swings", "Median", "Min", "Max"],
                   tablefmt="github"))
    print()

    # 2. Transition matrix
    print("-" * 40)
    print("2. TRANSITION MATRIX (probabilities)")
    print("-" * 40)
    all_states = [UP, DOWN, REDIST]
    header = ["From \\ To"] + [short(s) for s in all_states]
    mat_rows = []
    for src in all_states:
        row = [short(src)]
        for dst in all_states:
            p = pa["transition_probabilities"].get((src, dst), 0)
            row.append(f"{p:.2%}" if p else "—")
        mat_rows.append(row)
    print(tabulate(mat_rows, headers=header, tablefmt="github"))
    print()

    # Transition counts
    print("Raw transition counts:")
    for (src, dst), cnt in sorted(pa["transitions"].items(), key=lambda x: -x[1]):
        print(f"  {short(src)} → {short(dst)} : {cnt}")
    print()

    # 3. Top recurring paths
    print("-" * 40)
    print("3. TOP RECURRING PATHS (length 3)")
    print("-" * 40)
    for pat, cnt in pa["path_patterns_3"].items():
        label = " → ".join(short(s) for s in pat)
        print(f"  {label} : {cnt}")
    print()

    if pa["path_patterns_4"]:
        print("TOP RECURRING PATHS (length 4)")
        for pat, cnt in pa["path_patterns_4"].items():
            label = " → ".join(short(s) for s in pat)
            print(f"  {label} : {cnt}")
        print()

    # 4. Avg length
    print("-" * 40)
    print("4. AVG LENGTH PER STATE")
    print("-" * 40)
    for state in [UP, DOWN, REDIST]:
        if state in stats:
            s = stats[state]
            print(f"  {short(state):12s}: avg={s['avg_swings']:.1f}  median={s['median_swings']:.1f}  "
                  f"range=[{s['min_swings']}, {s['max_swings']}]  n={s['count']}")
    print()

    # 5. Repetition patterns
    print("-" * 40)
    print("5. REPETITION PATTERNS (reversal analysis)")
    print("-" * 40)
    for label, cnt in sorted(pa["repetition_patterns"].items(), key=lambda x: -x[1]):
        print(f"  {label} : {cnt}")
    print()

    # 6. RSI relational behavior
    print("-" * 40)
    print("6. RSI RELATIONAL BEHAVIOR PER STATE")
    print("-" * 40)
    for state in [UP, DOWN, REDIST]:
        if state in rsi_pat:
            r = rsi_pat[state]
            print(f"\n  [{short(state)}]")
            if "high_rising_pct" in r:
                print(f"    Swing Highs — RSI rising: {r['high_rising_pct']}%  "
                      f"falling: {r['high_falling_pct']}%  (n={r['high_sample_size']})")
            if "low_rising_pct" in r:
                print(f"    Swing Lows  — RSI rising: {r['low_rising_pct']}%  "
                      f"falling: {r['low_falling_pct']}%  (n={r['low_sample_size']})")
    print()

    # 7. Example segments
    print("-" * 40)
    print("7. EXAMPLE SEGMENTS")
    print("-" * 40)
    for state in [UP, DOWN, REDIST]:
        if state in examples:
            print(f"\n  [{short(state)}]")
            for i, ex in enumerate(examples[state], 1):
                print(f"    Example {i}: {ex['start_dt']} → {ex['end_dt']}  "
                      f"swings={ex['swing_count']}  labels={ex['swing_labels']}")
    print()
    print("=" * 72)
    print("  END OF REPORT")
    print("=" * 72)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Fetching DOGE/USDT 15m data (last 12 months)...")
    df = fetch_ohlcv("DOGE/USDT", "15m", months=12)
    print(f"  Fetched {len(df)} candles.")

    gaps = check_missing_candles(df)
    if len(gaps) > 0:
        print(f"  WARNING: {len(gaps)} gaps detected (> 22.5 min between candles)")
    else:
        print("  No missing candles detected.")

    # RSI
    print("Computing RSI (14, Wilder)...")
    df["rsi"] = wilder_rsi(df["close"], 14)

    # Swings
    print("Detecting swings...")
    swings = detect_swings(df)
    print(f"  Found {len(swings)} swings "
          f"({sum(1 for s in swings if s['type']=='high')} highs, "
          f"{sum(1 for s in swings if s['type']=='low')} lows)")

    # Label micro states
    print("Labelling micro states...")
    swings = label_micro_states(swings)

    # Build state path
    print("Building structural state path...")
    segments = build_state_path(swings)
    print(f"  Built {len(segments)} state segments.")

    # Path analysis
    print("Running path analysis...")
    pa = path_analysis(segments)

    # RSI behavior
    print("Analysing RSI relational behavior...")
    rsi_pat = rsi_behavior(segments)

    # Examples
    examples = pick_examples(segments, n=3)

    # Print report
    print()
    print_report(segments, pa, rsi_pat, examples, df, gaps)

    # Save JSON results
    json_out = {
        "data_summary": {
            "candles": len(df),
            "start": str(df["datetime"].iloc[0]),
            "end": str(df["datetime"].iloc[-1]),
            "gaps": len(gaps),
        },
        "swing_count": len(swings),
        "segment_count": len(segments),
        "state_counts": pa["length_stats"],
        "transitions": {f"{short(k[0])} -> {short(k[1])}": v
                        for k, v in pa["transitions"].items()},
        "transition_probabilities": {f"{short(k[0])} -> {short(k[1])}": v
                                     for k, v in pa["transition_probabilities"].items()},
        "path_patterns_3": {" -> ".join(short(s) for s in k): v
                            for k, v in pa["path_patterns_3"].items()},
        "path_patterns_4": {" -> ".join(short(s) for s in k): v
                            for k, v in pa["path_patterns_4"].items()},
        "repetition_patterns": pa["repetition_patterns"],
        "rsi_behavior": rsi_pat,
        "examples": {short(k): v for k, v in examples.items()},
    }
    out_path = "data/doge_structure_report.json"
    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\nJSON report saved to {out_path}")


if __name__ == "__main__":
    main()
