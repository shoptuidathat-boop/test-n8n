#!/usr/bin/env python3
"""
DOGE/USDT 15m Market Structure Analysis
========================================
Discovers recurring paths of market behavior:
  Continuation ↔ Redistribution ("state path" / "lối mòn")

Steps:
  1. Fetch 6-12 months of DOGE/USDT 15m candles from Binance
  2. Detect swing highs / swing lows (3-bar fractal)
  3. Label micro states (HH, HL, LH, LL)
  4. Classify structural states (CONTINUATION_UP, CONTINUATION_DOWN, REDISTRIBUTION)
  5. Build state path sequence
  6. Analyze path transitions, patterns, lengths, repetitions
  7. RSI relational behavior per state
"""

import json
import time
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. DATA FETCHING
# ---------------------------------------------------------------------------

SYMBOL = "DOGEUSDT"
INTERVAL = "15m"
BINANCE_KLINE_URL = "https://api.binance.us/api/v3/klines"
MS_15M = 15 * 60 * 1000


def fetch_klines(symbol: str, interval: str, months: int = 12) -> pd.DataFrame:
    """Fetch historical klines from Binance (paginated, 1000 per request)."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - months * 30 * 24 * 60 * 60 * 1000  # approx months

    all_rows = []
    current_start = start_ms

    while current_start < now_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_KLINE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        # next batch starts after last candle
        current_start = data[-1][0] + MS_15M
        time.sleep(0.15)  # rate-limit courtesy

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )
    df = df.drop_duplicates(subset=["open_time"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. RSI (Wilder, 14)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


# ---------------------------------------------------------------------------
# 3. SWING DETECTION (3-bar fractal)
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame):
    """
    Swing Low:  low[i] < low[i-1] AND low[i] < low[i+1]
    Swing High: high[i] > high[i-1] AND high[i] > high[i+1]

    Returns ordered list of dicts: {index, type, price, time, rsi}
    """
    highs = df["high"].values
    lows = df["low"].values
    rsi_vals = df["rsi"].values
    times = df["open_time"].values

    swings = []
    for i in range(1, len(df) - 1):
        is_swing_low = lows[i] < lows[i - 1] and lows[i] < lows[i + 1]
        is_swing_high = highs[i] > highs[i - 1] and highs[i] > highs[i + 1]

        if is_swing_high:
            swings.append({
                "index": i,
                "type": "high",
                "price": highs[i],
                "time": times[i],
                "rsi": rsi_vals[i],
            })
        if is_swing_low:
            swings.append({
                "index": i,
                "type": "low",
                "price": lows[i],
                "time": times[i],
                "rsi": rsi_vals[i],
            })

    # sort by index (time order)
    swings.sort(key=lambda s: (s["index"], 0 if s["type"] == "low" else 1))
    return swings


# ---------------------------------------------------------------------------
# 4. MICRO STATE LABELING (HH, HL, LH, LL)
# ---------------------------------------------------------------------------

def label_micro_states(swings: list) -> list:
    """
    Compare each swing to the previous swing of the same type.
    Highs: HH or LH
    Lows:  HL or LL
    """
    last_high = None
    last_low = None

    for sw in swings:
        if sw["type"] == "high":
            if last_high is not None:
                sw["micro"] = "HH" if sw["price"] > last_high["price"] else "LH"
            else:
                sw["micro"] = None  # first swing, no comparison
            last_high = sw
        else:  # low
            if last_low is not None:
                sw["micro"] = "HL" if sw["price"] > last_low["price"] else "LL"
            else:
                sw["micro"] = None
            last_low = sw

    return swings


# ---------------------------------------------------------------------------
# 5. STRUCTURAL STATE CLASSIFICATION
# ---------------------------------------------------------------------------

STATE_CONT_UP = "CONTINUATION_UP"
STATE_CONT_DOWN = "CONTINUATION_DOWN"
STATE_REDIST = "REDISTRIBUTION"


def classify_structural_states(swings: list) -> list:
    """
    CONTINUATION_UP:   HH + HL sequence
    CONTINUATION_DOWN: LH + LL sequence
    REDISTRIBUTION:    structural break confirmed by next swing

    Returns list of state segments:
      {state, start_idx, end_idx, start_swing, end_swing, swing_count, swings}
    """
    # Filter swings that have micro labels
    labeled = [s for s in swings if s.get("micro") is not None]
    if len(labeled) < 2:
        return []

    # Track the latest high-micro and low-micro
    current_high_micro = None
    current_low_micro = None
    current_state = None
    segments = []
    segment_swings = []

    def flush_segment(end_swing):
        nonlocal segment_swings
        if current_state and segment_swings:
            segments.append({
                "state": current_state,
                "start_idx": segment_swings[0]["index"],
                "end_idx": end_swing["index"],
                "start_time": segment_swings[0]["time"],
                "end_time": end_swing["time"],
                "swing_count": len(segment_swings),
                "swings": list(segment_swings),
            })
            segment_swings = []

    for sw in labeled:
        micro = sw["micro"]

        if sw["type"] == "high":
            current_high_micro = micro
        else:
            current_low_micro = micro

        # Determine what the current swing pair suggests
        if current_high_micro in ("HH",) and current_low_micro in ("HL",):
            implied = STATE_CONT_UP
        elif current_high_micro in ("LH",) and current_low_micro in ("LL",):
            implied = STATE_CONT_DOWN
        elif current_high_micro is None or current_low_micro is None:
            # Not enough data yet
            segment_swings.append(sw)
            continue
        else:
            implied = STATE_REDIST

        if current_state is None:
            current_state = implied
            segment_swings.append(sw)
        elif implied == current_state:
            segment_swings.append(sw)
        else:
            # State changed
            flush_segment(sw)
            current_state = implied
            segment_swings = [sw]

    # Flush last segment
    if segment_swings and current_state:
        segments.append({
            "state": current_state,
            "start_idx": segment_swings[0]["index"],
            "end_idx": segment_swings[-1]["index"],
            "start_time": segment_swings[0]["time"],
            "end_time": segment_swings[-1]["time"],
            "swing_count": len(segment_swings),
            "swings": list(segment_swings),
        })

    return segments


# ---------------------------------------------------------------------------
# 6. STATE PATH + PATH ANALYSIS
# ---------------------------------------------------------------------------

def build_state_path(segments: list) -> list:
    """Convert segments to a concise state sequence."""
    return [
        {
            "state": seg["state"],
            "start_idx": seg["start_idx"],
            "end_idx": seg["end_idx"],
            "start_time": str(seg["start_time"]),
            "end_time": str(seg["end_time"]),
            "swing_count": seg["swing_count"],
        }
        for seg in segments
    ]


def analyze_paths(segments: list) -> dict:
    """
    Compute:
      1. Transition frequencies & probabilities
      2. Typical path patterns (2-gram, 3-gram)
      3. Length analysis (avg swings per state type)
      4. Repetition patterns (failed vs true reversal)
    """
    states = [seg["state"] for seg in segments]
    n = len(states)

    # 1. Transition matrix
    transition_counts = Counter()
    for i in range(n - 1):
        transition_counts[(states[i], states[i + 1])] += 1

    total_transitions = sum(transition_counts.values())
    transition_probs = {}
    # Also compute row-wise probabilities
    from_counts = Counter()
    for (fr, to), c in transition_counts.items():
        from_counts[fr] += c
    for (fr, to), c in transition_counts.items():
        transition_probs[f"{fr} → {to}"] = {
            "count": c,
            "prob_from_state": round(c / from_counts[fr], 4) if from_counts[fr] else 0,
            "prob_global": round(c / total_transitions, 4) if total_transitions else 0,
        }

    # 2. N-gram patterns
    bigrams = Counter()
    trigrams = Counter()
    for i in range(n - 1):
        bigrams[f"{states[i]} → {states[i+1]}"] += 1
    for i in range(n - 2):
        trigrams[f"{states[i]} → {states[i+1]} → {states[i+2]}"] += 1

    # 3. Length analysis
    length_by_state = defaultdict(list)
    for seg in segments:
        length_by_state[seg["state"]].append(seg["swing_count"])

    length_stats = {}
    for state, lengths in length_by_state.items():
        length_stats[state] = {
            "count": len(lengths),
            "avg_swings": round(np.mean(lengths), 2),
            "median_swings": round(float(np.median(lengths)), 2),
            "min_swings": int(np.min(lengths)),
            "max_swings": int(np.max(lengths)),
            "std_swings": round(float(np.std(lengths)), 2),
        }

    # 4. Repetition / reversal patterns
    # DOWN → REDIST → DOWN = failed reversal
    # DOWN → REDIST → UP   = true reversal
    # UP → REDIST → UP     = failed reversal
    # UP → REDIST → DOWN   = true reversal
    reversal_patterns = Counter()
    for i in range(n - 2):
        pattern = f"{states[i]} → {states[i+1]} → {states[i+2]}"
        reversal_patterns[pattern] += 1

    return {
        "transition_matrix": dict(
            sorted(transition_probs.items(), key=lambda x: -x[1]["count"])
        ),
        "bigram_patterns": dict(bigrams.most_common()),
        "trigram_patterns": dict(trigrams.most_common()),
        "length_stats": length_stats,
        "reversal_patterns": dict(reversal_patterns.most_common()),
        "total_transitions": total_transitions,
    }


# ---------------------------------------------------------------------------
# 7. RSI RELATIONAL BEHAVIOR
# ---------------------------------------------------------------------------

def analyze_rsi_behavior(segments: list) -> dict:
    """
    For each state: record RSI_high_n vs RSI_high_{n-1}, RSI_low_n vs RSI_low_{n-1}
    Output RSI patterns per state.
    """
    rsi_patterns = defaultdict(lambda: {
        "high_rising": 0, "high_falling": 0, "high_flat": 0,
        "low_rising": 0, "low_falling": 0, "low_flat": 0,
        "divergence_bearish": 0,  # price HH but RSI lower
        "divergence_bullish": 0,  # price LL but RSI higher
        "convergence": 0,
        "total_high_comparisons": 0,
        "total_low_comparisons": 0,
    })

    for seg in segments:
        state = seg["state"]
        highs_in_seg = [s for s in seg["swings"] if s["type"] == "high"]
        lows_in_seg = [s for s in seg["swings"] if s["type"] == "low"]

        # High RSI comparisons
        for j in range(1, len(highs_in_seg)):
            prev_rsi = highs_in_seg[j - 1]["rsi"]
            curr_rsi = highs_in_seg[j]["rsi"]
            prev_price = highs_in_seg[j - 1]["price"]
            curr_price = highs_in_seg[j]["price"]

            if pd.isna(prev_rsi) or pd.isna(curr_rsi):
                continue

            rsi_patterns[state]["total_high_comparisons"] += 1
            if curr_rsi > prev_rsi:
                rsi_patterns[state]["high_rising"] += 1
            elif curr_rsi < prev_rsi:
                rsi_patterns[state]["high_falling"] += 1
            else:
                rsi_patterns[state]["high_flat"] += 1

            # Bearish divergence: price higher, RSI lower
            if curr_price > prev_price and curr_rsi < prev_rsi:
                rsi_patterns[state]["divergence_bearish"] += 1
            # Convergence: both move same direction
            elif (curr_price > prev_price and curr_rsi > prev_rsi) or \
                 (curr_price < prev_price and curr_rsi < prev_rsi):
                rsi_patterns[state]["convergence"] += 1

        # Low RSI comparisons
        for j in range(1, len(lows_in_seg)):
            prev_rsi = lows_in_seg[j - 1]["rsi"]
            curr_rsi = lows_in_seg[j]["rsi"]
            prev_price = lows_in_seg[j - 1]["price"]
            curr_price = lows_in_seg[j]["price"]

            if pd.isna(prev_rsi) or pd.isna(curr_rsi):
                continue

            rsi_patterns[state]["total_low_comparisons"] += 1
            if curr_rsi > prev_rsi:
                rsi_patterns[state]["low_rising"] += 1
            elif curr_rsi < prev_rsi:
                rsi_patterns[state]["low_falling"] += 1
            else:
                rsi_patterns[state]["low_flat"] += 1

            # Bullish divergence: price lower, RSI higher
            if curr_price < prev_price and curr_rsi > prev_rsi:
                rsi_patterns[state]["divergence_bullish"] += 1
            elif (curr_price > prev_price and curr_rsi > prev_rsi) or \
                 (curr_price < prev_price and curr_rsi < prev_rsi):
                rsi_patterns[state]["convergence"] += 1

    # Convert to plain dict and add ratios
    result = {}
    for state, data in rsi_patterns.items():
        d = dict(data)
        th = d["total_high_comparisons"]
        tl = d["total_low_comparisons"]
        if th > 0:
            d["high_rising_pct"] = round(d["high_rising"] / th * 100, 1)
            d["high_falling_pct"] = round(d["high_falling"] / th * 100, 1)
        if tl > 0:
            d["low_rising_pct"] = round(d["low_rising"] / tl * 100, 1)
            d["low_falling_pct"] = round(d["low_falling"] / tl * 100, 1)
        result[state] = d

    return result


# ---------------------------------------------------------------------------
# 8. EXAMPLE SEGMENTS
# ---------------------------------------------------------------------------

def get_example_segments(segments: list, path_analysis: dict, max_examples: int = 3) -> dict:
    """Pick example segments for top recurring trigram paths."""
    examples = {}
    trigrams = path_analysis["trigram_patterns"]
    states_list = [seg["state"] for seg in segments]

    for pattern_str in list(trigrams.keys())[:5]:
        parts = [p.strip() for p in pattern_str.split("→")]
        if len(parts) != 3:
            continue
        found = []
        for i in range(len(states_list) - 2):
            if states_list[i] == parts[0] and states_list[i+1] == parts[1] and states_list[i+2] == parts[2]:
                example = []
                for j in range(3):
                    seg = segments[i + j]
                    example.append({
                        "state": seg["state"],
                        "start_time": str(seg["start_time"]),
                        "end_time": str(seg["end_time"]),
                        "swing_count": seg["swing_count"],
                    })
                found.append(example)
                if len(found) >= max_examples:
                    break
        if found:
            examples[pattern_str] = found

    return examples


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def format_report(
    df: pd.DataFrame,
    swings: list,
    segments: list,
    path: list,
    path_analysis: dict,
    rsi_behavior: dict,
    examples: dict,
) -> str:
    """Build human-readable report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  DOGE/USDT 15m — MARKET STRUCTURE PATH ANALYSIS")
    lines.append("  Continuation ↔ Redistribution ('lối mòn')")
    lines.append("=" * 72)
    lines.append("")

    # Data summary
    lines.append(f"Data range : {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
    lines.append(f"Candles    : {len(df):,}")
    lines.append(f"Swings     : {len(swings):,}  (highs: {sum(1 for s in swings if s['type']=='high'):,}, "
                 f"lows: {sum(1 for s in swings if s['type']=='low'):,})")
    lines.append(f"Segments   : {len(segments):,}")
    lines.append("")

    # --- 1. State Counts ---
    lines.append("-" * 72)
    lines.append("1. STATE COUNTS")
    lines.append("-" * 72)
    ls = path_analysis["length_stats"]
    for state, stats in sorted(ls.items()):
        lines.append(f"  {state:25s}  count={stats['count']:4d}  "
                     f"avg_swings={stats['avg_swings']:.1f}  "
                     f"median={stats['median_swings']:.0f}  "
                     f"range=[{stats['min_swings']}, {stats['max_swings']}]")
    lines.append("")

    # --- 2. Transition Matrix ---
    lines.append("-" * 72)
    lines.append("2. TRANSITION MATRIX")
    lines.append("-" * 72)
    lines.append(f"  Total transitions: {path_analysis['total_transitions']}")
    lines.append("")
    lines.append(f"  {'Transition':55s} {'Count':>6s}  {'P(from)':>8s}  {'P(global)':>9s}")
    lines.append(f"  {'-'*55} {'-'*6}  {'-'*8}  {'-'*9}")
    for trans, data in path_analysis["transition_matrix"].items():
        lines.append(f"  {trans:55s} {data['count']:6d}  "
                     f"{data['prob_from_state']:8.2%}  {data['prob_global']:9.2%}")
    lines.append("")

    # --- 3. Top Recurring Paths ---
    lines.append("-" * 72)
    lines.append("3. TOP RECURRING PATHS (trigrams)")
    lines.append("-" * 72)
    for pattern, count in list(path_analysis["trigram_patterns"].items())[:10]:
        lines.append(f"  {count:4d}x  {pattern}")
    lines.append("")

    lines.append("   Bigrams:")
    for pattern, count in list(path_analysis["bigram_patterns"].items())[:10]:
        lines.append(f"  {count:4d}x  {pattern}")
    lines.append("")

    # --- 4. Average Length ---
    lines.append("-" * 72)
    lines.append("4. AVERAGE LENGTH (swings per state)")
    lines.append("-" * 72)
    for state, stats in sorted(ls.items()):
        lines.append(f"  {state:25s}  avg={stats['avg_swings']:.2f}  "
                     f"std={stats['std_swings']:.2f}  "
                     f"median={stats['median_swings']:.0f}")
    lines.append("")

    # --- 5. RSI Behavior ---
    lines.append("-" * 72)
    lines.append("5. RSI RELATIONAL BEHAVIOR PER STATE")
    lines.append("-" * 72)
    for state, data in sorted(rsi_behavior.items()):
        lines.append(f"\n  [{state}]")
        lines.append(f"    High swings: {data['total_high_comparisons']} comparisons")
        if data["total_high_comparisons"] > 0:
            lines.append(f"      RSI rising  : {data['high_rising']:4d} ({data.get('high_rising_pct',0):.1f}%)")
            lines.append(f"      RSI falling : {data['high_falling']:4d} ({data.get('high_falling_pct',0):.1f}%)")
        lines.append(f"    Low swings: {data['total_low_comparisons']} comparisons")
        if data["total_low_comparisons"] > 0:
            lines.append(f"      RSI rising  : {data['low_rising']:4d} ({data.get('low_rising_pct',0):.1f}%)")
            lines.append(f"      RSI falling : {data['low_falling']:4d} ({data.get('low_falling_pct',0):.1f}%)")
        lines.append(f"    Bearish divergence : {data['divergence_bearish']}")
        lines.append(f"    Bullish divergence : {data['divergence_bullish']}")
        lines.append(f"    Convergence        : {data['convergence']}")
    lines.append("")

    # --- 6. Example Segments ---
    lines.append("-" * 72)
    lines.append("6. EXAMPLE SEGMENTS FOR TOP PATHS")
    lines.append("-" * 72)
    for pattern, exs in examples.items():
        lines.append(f"\n  Pattern: {pattern}  ({len(exs)} example(s))")
        for idx, ex in enumerate(exs):
            lines.append(f"    Example {idx+1}:")
            for seg in ex:
                lines.append(f"      {seg['state']:25s}  {seg['start_time']} → {seg['end_time']}  "
                             f"({seg['swing_count']} swings)")
    lines.append("")

    # --- 7. Reversal / Failed Reversal ---
    lines.append("-" * 72)
    lines.append("7. REVERSAL vs FAILED REVERSAL PATTERNS")
    lines.append("-" * 72)
    rp = path_analysis["reversal_patterns"]
    for pattern, count in list(rp.items())[:10]:
        tag = ""
        parts = [p.strip() for p in pattern.split("→")]
        if len(parts) == 3 and parts[1] == STATE_REDIST:
            if parts[0] == parts[2]:
                tag = " ← FAILED REVERSAL"
            elif parts[0] != parts[2]:
                tag = " ← TRUE REVERSAL"
        lines.append(f"  {count:4d}x  {pattern}{tag}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    print("Fetching DOGE/USDT 15m data from Binance (last 12 months)...")
    df = fetch_klines(SYMBOL, INTERVAL, months=12)
    print(f"  → {len(df):,} candles  [{df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}]")

    # Check for missing candles
    expected_gap = pd.Timedelta(minutes=15)
    gaps = df["open_time"].diff().dropna()
    n_missing = (gaps > expected_gap * 1.5).sum()
    print(f"  → Missing candle gaps: {n_missing}")

    # RSI
    print("Computing RSI(14, Wilder)...")
    df["rsi"] = compute_rsi(df["close"], period=14)

    # Swing detection
    print("Detecting swings...")
    swings = detect_swings(df)
    print(f"  → {len(swings)} swings detected")

    # Micro state labeling
    print("Labeling micro states...")
    swings = label_micro_states(swings)

    # Structural state classification
    print("Classifying structural states...")
    segments = classify_structural_states(swings)
    print(f"  → {len(segments)} structural segments")

    # State path
    print("Building state path...")
    state_path = build_state_path(segments)

    # Path analysis
    print("Analyzing paths...")
    path_analysis = analyze_paths(segments)

    # RSI behavior
    print("Analyzing RSI relational behavior...")
    rsi_behavior = analyze_rsi_behavior(segments)

    # Examples
    print("Collecting example segments...")
    examples = get_example_segments(segments, path_analysis)

    # Report
    report = format_report(df, swings, segments, state_path, path_analysis, rsi_behavior, examples)
    print("\n" + report)

    # Save outputs
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    report_path = out_dir / "market_structure_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    # Save JSON data for programmatic use
    json_out = {
        "metadata": {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "candles": len(df),
            "date_range": f"{df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}",
            "missing_gaps": int(n_missing),
            "total_swings": len(swings),
            "total_segments": len(segments),
        },
        "state_path": state_path,
        "path_analysis": path_analysis,
        "rsi_behavior": rsi_behavior,
        "examples": examples,
    }
    json_path = out_dir / "market_structure_data.json"
    json_path.write_text(json.dumps(json_out, indent=2, default=str), encoding="utf-8")
    print(f"JSON data saved to {json_path}")


if __name__ == "__main__":
    main()
