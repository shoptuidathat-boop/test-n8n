#!/usr/bin/env python3
"""
DOGE/USDT 15m Market Structure Analysis
========================================
Discovers recurring paths of market behavior:
  Continuation <-> Redistribution ("state path" / "lối mòn")

Methodology:
  - Swing detection via fractal logic
  - Micro state labeling (HH, LH, HL, LL)
  - Structural state classification (CONTINUATION_UP, CONTINUATION_DOWN, REDISTRIBUTION)
  - State path construction & pattern analysis
  - Relational RSI behavior per state
"""

import time
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
from tabulate import tabulate


# ---------------------------------------------------------------------------
# 0. DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str = "DOGE/USDT", timeframe: str = "15m",
                months: int = 12) -> pd.DataFrame:
    """Fetch OHLCV data from Binance. Paginate to cover *months* of history."""
    exchange = ccxt.binance({"enableRateLimit": True})

    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    now = exchange.milliseconds()
    since = now - months * 30 * 24 * 60 * 60 * 1000  # approximate

    all_candles: list[list] = []
    cursor = since

    while cursor < now:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        cursor = candles[-1][0] + ms_per_candle
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def check_missing_candles(df: pd.DataFrame, timeframe_ms: int = 15 * 60 * 1000) -> dict:
    """Report gaps in the candle series."""
    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > timeframe_ms * 1.5]
    return {
        "total_candles": len(df),
        "expected_gap_ms": timeframe_ms,
        "missing_segments": len(gaps),
        "largest_gap_minutes": int(gaps.max() / 60_000) if len(gaps) else 0,
    }


# ---------------------------------------------------------------------------
# 1. RSI (Wilder, period=14)
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# 2. SWING DETECTION (fractal 1)
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame) -> list[dict]:
    """
    Swing Low:  low[i] < low[i-1] AND low[i] < low[i+1]
    Swing High: high[i] > high[i-1] AND high[i] > high[i+1]
    Returns ordered list of swings.
    """
    highs = df["high"].values
    lows = df["low"].values
    rsi = df["rsi"].values
    timestamps = df["timestamp"].values

    swings: list[dict] = []

    for i in range(1, len(df) - 1):
        is_swing_low = lows[i] < lows[i - 1] and lows[i] < lows[i + 1]
        is_swing_high = highs[i] > highs[i - 1] and highs[i] > highs[i + 1]

        if is_swing_low:
            swings.append({
                "index": i,
                "type": "low",
                "price": float(lows[i]),
                "rsi": float(rsi[i]),
                "timestamp": int(timestamps[i]),
            })
        if is_swing_high:
            swings.append({
                "index": i,
                "type": "high",
                "price": float(highs[i]),
                "rsi": float(rsi[i]),
                "timestamp": int(timestamps[i]),
            })

    swings.sort(key=lambda s: s["index"])
    return swings


# ---------------------------------------------------------------------------
# STEP 1 — LABEL MICRO STATE PER SWING
# ---------------------------------------------------------------------------

def label_micro_states(swings: list[dict]) -> list[dict]:
    """
    Compare each swing to the previous swing of the same type.
    Highs: HH / LH
    Lows:  HL / LL
    """
    prev_high: Optional[dict] = None
    prev_low: Optional[dict] = None

    for sw in swings:
        if sw["type"] == "high":
            if prev_high is not None:
                sw["micro"] = "HH" if sw["price"] > prev_high["price"] else "LH"
                sw["rsi_prev"] = prev_high["rsi"]
            else:
                sw["micro"] = None
                sw["rsi_prev"] = None
            prev_high = sw
        else:  # low
            if prev_low is not None:
                sw["micro"] = "HL" if sw["price"] > prev_low["price"] else "LL"
                sw["rsi_prev"] = prev_low["rsi"]
            else:
                sw["micro"] = None
                sw["rsi_prev"] = None
            prev_low = sw

    return swings


# ---------------------------------------------------------------------------
# STEP 2 — BUILD STRUCTURAL STATE
# ---------------------------------------------------------------------------

STATE_UP = "CONTINUATION_UP"
STATE_DOWN = "CONTINUATION_DOWN"
STATE_REDIST = "REDISTRIBUTION"


def classify_structural_states(swings: list[dict]) -> list[dict]:
    """
    At each step classify the current structural state.

    CONTINUATION_UP:   HH + HL sequence
    CONTINUATION_DOWN: LH + LL sequence
    REDISTRIBUTION:    structural break detected
        - from UP: HL -> LL
        - from DOWN: LH -> HH
      confirmed by next swing continuing opposite direction.
    """
    labeled = swings.copy()

    # Track the latest micro label for highs and lows separately
    last_high_micro: Optional[str] = None
    last_low_micro: Optional[str] = None
    current_state: Optional[str] = None

    for sw in labeled:
        if sw["micro"] is None:
            sw["state"] = None
            continue

        if sw["type"] == "high":
            last_high_micro = sw["micro"]
        else:
            last_low_micro = sw["micro"]

        # Need both a high and a low micro to classify
        if last_high_micro is None or last_low_micro is None:
            sw["state"] = None
            continue

        # Determine what the structure says right now
        if last_high_micro == "HH" and last_low_micro == "HL":
            new_state = STATE_UP
        elif last_high_micro == "LH" and last_low_micro == "LL":
            new_state = STATE_DOWN
        else:
            # Mixed signals → redistribution
            new_state = STATE_REDIST

        # Redistribution requires a structural break from a prior trend
        if current_state in (STATE_UP, STATE_DOWN) and new_state != current_state:
            new_state = STATE_REDIST

        current_state = new_state
        sw["state"] = current_state

    return labeled


# ---------------------------------------------------------------------------
# STEP 3 — BUILD STATE PATH
# ---------------------------------------------------------------------------

def build_state_path(swings: list[dict]) -> list[dict]:
    """
    Convert timeline into contiguous state segments.
    Each segment: state_type, start_index, end_index, swing_count
    """
    segments: list[dict] = []
    current_state: Optional[str] = None
    segment_swings: list[dict] = []

    for sw in swings:
        st = sw.get("state")
        if st is None:
            continue

        if st != current_state:
            # Close previous segment
            if current_state is not None and segment_swings:
                segments.append({
                    "state": current_state,
                    "start_index": segment_swings[0]["index"],
                    "end_index": segment_swings[-1]["index"],
                    "start_ts": segment_swings[0]["timestamp"],
                    "end_ts": segment_swings[-1]["timestamp"],
                    "swing_count": len(segment_swings),
                })
            current_state = st
            segment_swings = [sw]
        else:
            segment_swings.append(sw)

    # Close last segment
    if current_state is not None and segment_swings:
        segments.append({
            "state": current_state,
            "start_index": segment_swings[0]["index"],
            "end_index": segment_swings[-1]["index"],
            "start_ts": segment_swings[0]["timestamp"],
            "end_ts": segment_swings[-1]["timestamp"],
            "swing_count": len(segment_swings),
        })

    return segments


# ---------------------------------------------------------------------------
# STEP 4 — PATH ANALYSIS (LỐI MÒN)
# ---------------------------------------------------------------------------

def analyze_paths(segments: list[dict]) -> dict:
    """
    1. Frequency of transitions
    2. Typical path patterns (length-2 and length-3)
    3. Length analysis (avg swings per state type)
    4. Repetition patterns (failed vs true reversals)
    """
    # --- 1. Transition frequency ---
    transitions: Counter = Counter()
    for i in range(len(segments) - 1):
        pair = (segments[i]["state"], segments[i + 1]["state"])
        transitions[pair] += 1

    # --- 2. Path patterns (trigrams) ---
    trigrams: Counter = Counter()
    for i in range(len(segments) - 2):
        tri = (segments[i]["state"], segments[i + 1]["state"], segments[i + 2]["state"])
        trigrams[tri] += 1

    bigrams: Counter = Counter()
    for i in range(len(segments) - 1):
        bi = (segments[i]["state"], segments[i + 1]["state"])
        bigrams[bi] += 1

    # --- 3. Length analysis ---
    length_by_state: defaultdict[str, list[int]] = defaultdict(list)
    for seg in segments:
        length_by_state[seg["state"]].append(seg["swing_count"])

    avg_lengths = {
        state: round(np.mean(counts), 2)
        for state, counts in length_by_state.items()
    }
    median_lengths = {
        state: round(float(np.median(counts)), 2)
        for state, counts in length_by_state.items()
    }

    # --- 4. Reversal analysis ---
    reversal_stats: Counter = Counter()
    for i in range(len(segments) - 2):
        a, b, c = segments[i]["state"], segments[i + 1]["state"], segments[i + 2]["state"]
        if b == STATE_REDIST:
            if a == STATE_DOWN and c == STATE_UP:
                reversal_stats["DOWN→REDIST→UP (true reversal)"] += 1
            elif a == STATE_DOWN and c == STATE_DOWN:
                reversal_stats["DOWN→REDIST→DOWN (failed reversal)"] += 1
            elif a == STATE_UP and c == STATE_DOWN:
                reversal_stats["UP→REDIST→DOWN (true reversal)"] += 1
            elif a == STATE_UP and c == STATE_UP:
                reversal_stats["UP→REDIST→UP (failed reversal)"] += 1

    # --- Transition matrix (probabilities) ---
    state_names = [STATE_UP, STATE_DOWN, STATE_REDIST]
    trans_matrix: dict[str, dict[str, float]] = {}
    for src in state_names:
        total = sum(transitions.get((src, dst), 0) for dst in state_names)
        row: dict[str, float] = {}
        for dst in state_names:
            count = transitions.get((src, dst), 0)
            row[dst] = round(count / total, 3) if total else 0.0
        trans_matrix[src] = row

    return {
        "transitions": dict(transitions),
        "bigrams": dict(bigrams),
        "trigrams": dict(trigrams),
        "avg_lengths": avg_lengths,
        "median_lengths": median_lengths,
        "reversal_stats": dict(reversal_stats),
        "transition_matrix": trans_matrix,
    }


# ---------------------------------------------------------------------------
# STEP 5 — RSI BEHAVIOR (RELATIONAL)
# ---------------------------------------------------------------------------

def analyze_rsi_per_state(swings: list[dict], segments: list[dict]) -> dict:
    """
    For each state segment, record relational RSI:
      RSI_high_n vs RSI_high_{n-1}
      RSI_low_n  vs RSI_low_{n-1}
    Classify as RSI_RISING, RSI_FALLING, RSI_FLAT (< 2 pts difference)
    """
    # Index swings by candle index for quick lookup
    swing_lookup: dict[int, dict] = {sw["index"]: sw for sw in swings}

    state_rsi: defaultdict[str, list[dict]] = defaultdict(list)

    for seg in segments:
        seg_swings = [
            sw for sw in swings
            if seg["start_index"] <= sw["index"] <= seg["end_index"]
            and sw.get("micro") is not None
            and sw.get("rsi_prev") is not None
        ]
        for sw in seg_swings:
            rsi_delta = sw["rsi"] - sw["rsi_prev"]
            if abs(rsi_delta) < 2:
                direction = "FLAT"
            elif rsi_delta > 0:
                direction = "RISING"
            else:
                direction = "FALLING"

            state_rsi[seg["state"]].append({
                "swing_type": sw["type"],
                "micro": sw["micro"],
                "rsi_delta": round(rsi_delta, 2),
                "direction": direction,
            })

    # Summarize
    summary: dict[str, dict] = {}
    for state, entries in state_rsi.items():
        high_entries = [e for e in entries if e["swing_type"] == "high"]
        low_entries = [e for e in entries if e["swing_type"] == "low"]

        def _summarize(items: list[dict]) -> dict:
            if not items:
                return {"count": 0}
            dirs = Counter(e["direction"] for e in items)
            total = len(items)
            avg_delta = round(np.mean([e["rsi_delta"] for e in items]), 2)
            return {
                "count": total,
                "avg_rsi_delta": avg_delta,
                "pct_rising": round(dirs.get("RISING", 0) / total * 100, 1),
                "pct_falling": round(dirs.get("FALLING", 0) / total * 100, 1),
                "pct_flat": round(dirs.get("FLAT", 0) / total * 100, 1),
            }

        summary[state] = {
            "highs": _summarize(high_entries),
            "lows": _summarize(low_entries),
        }

    return summary


# ---------------------------------------------------------------------------
# OUTPUT FORMATTING
# ---------------------------------------------------------------------------

def _short(state: str) -> str:
    return state.replace("CONTINUATION_", "CONT_").replace("REDISTRIBUTION", "REDIST")


def format_output(df: pd.DataFrame, swings: list[dict], segments: list[dict],
                  path_analysis: dict, rsi_analysis: dict, gap_info: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  DOGE/USDT 15m — MARKET STRUCTURE PATH ANALYSIS")
    lines.append("=" * 72)

    # Data summary
    lines.append(f"\nData range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    lines.append(f"Total candles: {len(df):,}")
    lines.append(f"Missing segments: {gap_info['missing_segments']}  "
                 f"(largest gap: {gap_info['largest_gap_minutes']} min)")
    lines.append(f"Total swings detected: {len(swings)}")
    lines.append(f"Total state segments: {len(segments)}")

    # 1. State counts
    lines.append("\n" + "-" * 50)
    lines.append("1. STATE COUNTS")
    lines.append("-" * 50)
    state_counts = Counter(seg["state"] for seg in segments)
    for st in [STATE_UP, STATE_DOWN, STATE_REDIST]:
        lines.append(f"  {_short(st):20s} : {state_counts.get(st, 0)}")

    # 2. Transition matrix
    lines.append("\n" + "-" * 50)
    lines.append("2. TRANSITION MATRIX (probabilities)")
    lines.append("-" * 50)
    tm = path_analysis["transition_matrix"]
    header = ["From \\ To"] + [_short(s) for s in [STATE_UP, STATE_DOWN, STATE_REDIST]]
    rows = []
    for src in [STATE_UP, STATE_DOWN, STATE_REDIST]:
        row = [_short(src)]
        for dst in [STATE_UP, STATE_DOWN, STATE_REDIST]:
            row.append(f"{tm[src][dst]:.3f}")
        rows.append(row)
    lines.append(tabulate(rows, headers=header, tablefmt="simple"))

    # Raw counts
    lines.append("\n  Raw transition counts:")
    for (src, dst), cnt in sorted(path_analysis["transitions"].items(), key=lambda x: -x[1]):
        lines.append(f"    {_short(src):20s} → {_short(dst):20s} : {cnt}")

    # 3. Top recurring paths (trigrams)
    lines.append("\n" + "-" * 50)
    lines.append("3. TOP RECURRING PATHS (trigrams)")
    lines.append("-" * 50)
    sorted_tri = sorted(path_analysis["trigrams"].items(), key=lambda x: -x[1])
    for tri, cnt in sorted_tri[:15]:
        path_str = " → ".join(_short(s) for s in tri)
        lines.append(f"  {path_str:55s} : {cnt}")

    # Bigrams
    lines.append("\n  Top bigrams:")
    sorted_bi = sorted(path_analysis["bigrams"].items(), key=lambda x: -x[1])
    for bi, cnt in sorted_bi[:10]:
        path_str = " → ".join(_short(s) for s in bi)
        lines.append(f"  {path_str:40s} : {cnt}")

    # 4. Avg length
    lines.append("\n" + "-" * 50)
    lines.append("4. AVERAGE LENGTH (swings per state)")
    lines.append("-" * 50)
    for st in [STATE_UP, STATE_DOWN, STATE_REDIST]:
        avg = path_analysis["avg_lengths"].get(st, 0)
        med = path_analysis["median_lengths"].get(st, 0)
        lines.append(f"  {_short(st):20s} : avg={avg:5.2f}  median={med:5.2f}")

    # 5. Reversal analysis
    lines.append("\n" + "-" * 50)
    lines.append("5. REVERSAL PATTERNS")
    lines.append("-" * 50)
    for pattern, cnt in sorted(path_analysis["reversal_stats"].items(), key=lambda x: -x[1]):
        lines.append(f"  {pattern:50s} : {cnt}")

    # 6. RSI relational behavior
    lines.append("\n" + "-" * 50)
    lines.append("6. RSI RELATIONAL BEHAVIOR PER STATE")
    lines.append("-" * 50)
    for st in [STATE_UP, STATE_DOWN, STATE_REDIST]:
        info = rsi_analysis.get(st, {})
        lines.append(f"\n  [{_short(st)}]")
        for swing_type in ["highs", "lows"]:
            data = info.get(swing_type, {"count": 0})
            if data["count"] == 0:
                lines.append(f"    {swing_type:6s}: no data")
                continue
            lines.append(
                f"    {swing_type:6s}: n={data['count']:4d}  "
                f"avg_Δrsi={data['avg_rsi_delta']:+6.2f}  "
                f"rising={data['pct_rising']:5.1f}%  "
                f"falling={data['pct_falling']:5.1f}%  "
                f"flat={data['pct_flat']:5.1f}%"
            )

    # 7. Example segments
    lines.append("\n" + "-" * 50)
    lines.append("7. EXAMPLE SEGMENTS (first 3 of each state)")
    lines.append("-" * 50)
    examples_per_state: defaultdict[str, list] = defaultdict(list)
    for seg in segments:
        if len(examples_per_state[seg["state"]]) < 3:
            examples_per_state[seg["state"]].append(seg)

    for st in [STATE_UP, STATE_DOWN, STATE_REDIST]:
        lines.append(f"\n  [{_short(st)}]")
        for seg in examples_per_state.get(st, []):
            start_dt = datetime.fromtimestamp(seg["start_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            end_dt = datetime.fromtimestamp(seg["end_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"    candles [{seg['start_index']:>6d} – {seg['end_index']:>6d}]  "
                f"{start_dt} → {end_dt}  "
                f"swings={seg['swing_count']}"
            )

    # 8. Full state path sequence (abbreviated)
    lines.append("\n" + "-" * 50)
    lines.append("8. FULL STATE PATH SEQUENCE (abbreviated)")
    lines.append("-" * 50)
    abbrev = {"CONTINUATION_UP": "UP", "CONTINUATION_DOWN": "DN", "REDISTRIBUTION": "RD"}
    path_str = " → ".join(abbrev.get(seg["state"], "??") for seg in segments)
    # Wrap at ~80 chars
    wrapped: list[str] = []
    current_line = "  "
    for token in path_str.split(" "):
        if len(current_line) + len(token) + 1 > 78:
            wrapped.append(current_line)
            current_line = "  " + token
        else:
            current_line += " " + token if current_line.strip() else "  " + token
    if current_line.strip():
        wrapped.append(current_line)
    lines.append("\n".join(wrapped))

    lines.append("\n" + "=" * 72)
    lines.append("  END OF ANALYSIS")
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Fetching DOGE/USDT 15m data (last 12 months)...")
    df = fetch_ohlcv("DOGE/USDT", "15m", months=12)
    print(f"  → {len(df):,} candles fetched")

    gap_info = check_missing_candles(df)
    print(f"  → Missing segments: {gap_info['missing_segments']}")

    print("Computing RSI (14, Wilder)...")
    df["rsi"] = compute_rsi(df["close"], period=14)

    print("Detecting swings...")
    swings = detect_swings(df)
    print(f"  → {len(swings)} swings detected")

    print("Step 1: Labeling micro states...")
    swings = label_micro_states(swings)

    print("Step 2: Classifying structural states...")
    swings = classify_structural_states(swings)

    print("Step 3: Building state path...")
    segments = build_state_path(swings)
    print(f"  → {len(segments)} state segments")

    print("Step 4: Path analysis...")
    path_analysis = analyze_paths(segments)

    print("Step 5: RSI relational analysis...")
    rsi_analysis = analyze_rsi_per_state(swings, segments)

    print("Generating report...\n")
    report = format_output(df, swings, segments, path_analysis, rsi_analysis, gap_info)
    print(report)

    # Save report
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(output_dir, "data", "structure_analysis_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save raw data as JSON for further processing
    json_path = os.path.join(output_dir, "data", "structure_analysis_raw.json")
    raw_data = {
        "gap_info": gap_info,
        "total_swings": len(swings),
        "total_segments": len(segments),
        "segments": segments,
        "path_analysis": {
            "transitions": {f"{k[0]}→{k[1]}": v for k, v in path_analysis["transitions"].items()},
            "trigrams": {" → ".join(k): v for k, v in path_analysis["trigrams"].items()},
            "avg_lengths": path_analysis["avg_lengths"],
            "median_lengths": path_analysis["median_lengths"],
            "reversal_stats": path_analysis["reversal_stats"],
            "transition_matrix": path_analysis["transition_matrix"],
        },
        "rsi_analysis": rsi_analysis,
    }
    with open(json_path, "w") as f:
        json.dump(raw_data, f, indent=2, default=str)
    print(f"Raw data saved to: {json_path}")


if __name__ == "__main__":
    main()
