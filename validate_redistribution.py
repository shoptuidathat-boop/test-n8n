#!/usr/bin/env python3
"""
Redistribution Behavior Validation
====================================
Validates stability of redistribution behavior across:
  - Long time (2-4 years)
  - Multiple assets (DOGE, BTC, SOL)
  - Different swing definitions (Fractal 1 vs Fractal 2)

Focus: Behavior only (NOT prediction)
  1) REVERSAL
  2) CONTINUATION

See VALIDATION_SPEC.md for full methodology.
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


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_ohlcv(symbol: str, timeframe: str = "15m",
                years: int = 4) -> pd.DataFrame:
    """Fetch OHLCV data from Binance. Paginate to cover full history."""
    exchange = ccxt.binance({"enableRateLimit": True})

    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    now = exchange.milliseconds()
    since = now - years * 365 * 24 * 60 * 60 * 1000

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
    diffs = df["timestamp"].diff().dropna()
    gaps = diffs[diffs > timeframe_ms * 1.5]
    return {
        "total_candles": len(df),
        "missing_segments": len(gaps),
        "largest_gap_minutes": int(gaps.max() / 60_000) if len(gaps) else 0,
    }


# ============================================================================
# RSI (Wilder, period=14)
# ============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================================
# SWING DETECTION
# ============================================================================

def detect_swings_fractal1(df: pd.DataFrame) -> list[dict]:
    """Fractal 1: compare with 1 neighbor each side."""
    highs = df["high"].values
    lows = df["low"].values
    rsi = df["rsi"].values
    timestamps = df["timestamp"].values
    swings: list[dict] = []

    for i in range(1, len(df) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swings.append({"index": i, "type": "low", "price": float(lows[i]),
                           "rsi": float(rsi[i]), "timestamp": int(timestamps[i])})
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swings.append({"index": i, "type": "high", "price": float(highs[i]),
                           "rsi": float(rsi[i]), "timestamp": int(timestamps[i])})

    swings.sort(key=lambda s: s["index"])
    return swings


def detect_swings_fractal2(df: pd.DataFrame) -> list[dict]:
    """Fractal 2: compare with 2 neighbors each side."""
    highs = df["high"].values
    lows = df["low"].values
    rsi = df["rsi"].values
    timestamps = df["timestamp"].values
    swings: list[dict] = []

    for i in range(2, len(df) - 2):
        if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                lows[i] < lows[i + 1] and lows[i] < lows[i + 2]):
            swings.append({"index": i, "type": "low", "price": float(lows[i]),
                           "rsi": float(rsi[i]), "timestamp": int(timestamps[i])})
        if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                highs[i] > highs[i + 1] and highs[i] > highs[i + 2]):
            swings.append({"index": i, "type": "high", "price": float(highs[i]),
                           "rsi": float(rsi[i]), "timestamp": int(timestamps[i])})

    swings.sort(key=lambda s: s["index"])
    return swings


# ============================================================================
# STEP 1 — MICRO STATE LABELING
# ============================================================================

def label_micro_states(swings: list[dict]) -> list[dict]:
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
        else:
            if prev_low is not None:
                sw["micro"] = "HL" if sw["price"] > prev_low["price"] else "LL"
                sw["rsi_prev"] = prev_low["rsi"]
            else:
                sw["micro"] = None
                sw["rsi_prev"] = None
            prev_low = sw
    return swings


# ============================================================================
# STRUCTURAL STATE CLASSIFICATION
# ============================================================================

STATE_UP = "CONTINUATION_UP"
STATE_DOWN = "CONTINUATION_DOWN"
STATE_REDIST = "REDISTRIBUTION"


def classify_structural_states(swings: list[dict]) -> list[dict]:
    last_high_micro: Optional[str] = None
    last_low_micro: Optional[str] = None
    current_state: Optional[str] = None

    for sw in swings:
        if sw.get("micro") is None:
            sw["state"] = None
            continue

        if sw["type"] == "high":
            last_high_micro = sw["micro"]
        else:
            last_low_micro = sw["micro"]

        if last_high_micro is None or last_low_micro is None:
            sw["state"] = None
            continue

        if last_high_micro == "HH" and last_low_micro == "HL":
            new_state = STATE_UP
        elif last_high_micro == "LH" and last_low_micro == "LL":
            new_state = STATE_DOWN
        else:
            new_state = STATE_REDIST

        if current_state in (STATE_UP, STATE_DOWN) and new_state != current_state:
            new_state = STATE_REDIST

        current_state = new_state
        sw["state"] = current_state

    return swings


# ============================================================================
# STATE PATH CONSTRUCTION
# ============================================================================

def build_state_path(swings: list[dict]) -> list[dict]:
    segments: list[dict] = []
    current_state: Optional[str] = None
    segment_swings: list[dict] = []

    for sw in swings:
        st = sw.get("state")
        if st is None:
            continue
        if st != current_state:
            if current_state is not None and segment_swings:
                segments.append(_make_segment(current_state, segment_swings))
            current_state = st
            segment_swings = [sw]
        else:
            segment_swings.append(sw)

    if current_state is not None and segment_swings:
        segments.append(_make_segment(current_state, segment_swings))

    return segments


def _make_segment(state: str, seg_swings: list[dict]) -> dict:
    micros = [sw["micro"] for sw in seg_swings if sw.get("micro")]
    return {
        "state": state,
        "start_index": seg_swings[0]["index"],
        "end_index": seg_swings[-1]["index"],
        "start_ts": seg_swings[0]["timestamp"],
        "end_ts": seg_swings[-1]["timestamp"],
        "swing_count": len(seg_swings),
        "swings": seg_swings,
        "structure_sequence": micros,
    }


# ============================================================================
# STEP 2 — LABEL OUTCOME (REVERSAL / CONTINUATION)
# ============================================================================

def label_redistribution_outcomes(segments: list[dict]) -> list[dict]:
    """For each REDISTRIBUTION segment, label its outcome."""
    redist_segments: list[dict] = []

    for i, seg in enumerate(segments):
        if seg["state"] != STATE_REDIST:
            continue

        # Find the state before this redistribution
        prev_state = None
        for j in range(i - 1, -1, -1):
            if segments[j]["state"] in (STATE_UP, STATE_DOWN):
                prev_state = segments[j]["state"]
                break

        # Find the state after this redistribution
        next_state = None
        for j in range(i + 1, len(segments)):
            if segments[j]["state"] in (STATE_UP, STATE_DOWN):
                next_state = segments[j]["state"]
                break

        if prev_state is None or next_state is None:
            outcome = "UNKNOWN"
        elif prev_state != next_state:
            outcome = "REVERSAL"
        else:
            outcome = "CONTINUATION"

        seg_copy = dict(seg)
        seg_copy["prev_state"] = prev_state
        seg_copy["next_state"] = next_state
        seg_copy["outcome"] = outcome
        redist_segments.append(seg_copy)

    return redist_segments


# ============================================================================
# STEP 3 — CORE BEHAVIOR FEATURES
# ============================================================================

def compute_behavior_features(redist_segments: list[dict]) -> list[dict]:
    """Compute features for each redistribution segment."""
    for seg in redist_segments:
        micros = seg["structure_sequence"]
        seg["num_swings"] = seg["swing_count"]

        # Balance
        up_count = sum(1 for m in micros if m in ("HH", "HL"))
        down_count = sum(1 for m in micros if m in ("LH", "LL"))
        if up_count > down_count:
            seg["balance"] = "UP"
        elif down_count > up_count:
            seg["balance"] = "DOWN"
        else:
            seg["balance"] = "EQUAL"

        # RSI relational sequences
        rsi_high_seq: list[str] = []
        rsi_low_seq: list[str] = []
        for sw in seg["swings"]:
            if sw.get("rsi_prev") is None:
                continue
            delta = sw["rsi"] - sw["rsi_prev"]
            if abs(delta) < 2:
                d = "="
            elif delta > 0:
                d = ">"
            else:
                d = "<"

            if sw["type"] == "high":
                rsi_high_seq.append(d)
            else:
                rsi_low_seq.append(d)

        seg["rsi_high_seq"] = rsi_high_seq
        seg["rsi_low_seq"] = rsi_low_seq
        seg["has_rsi_sequence"] = (len(rsi_high_seq) >= 2 or len(rsi_low_seq) >= 2)

    return redist_segments


# ============================================================================
# STEP 5 — CORE METRICS
# ============================================================================

def compute_metrics(redist_segments: list[dict]) -> dict:
    """Compute all core metrics from labeled redistribution segments."""
    valid = [s for s in redist_segments if s["outcome"] in ("REVERSAL", "CONTINUATION")]
    if not valid:
        return {"error": "no valid segments"}

    total = len(valid)
    reversals = sum(1 for s in valid if s["outcome"] == "REVERSAL")
    continuations = total - reversals

    # Distribution
    pct_reversal = round(reversals / total * 100, 2)
    pct_continuation = round(continuations / total * 100, 2)

    # Length effect
    short = [s for s in valid if s["num_swings"] <= 2]
    long_ = [s for s in valid if s["num_swings"] >= 3]
    p_rev_short = round(sum(1 for s in short if s["outcome"] == "REVERSAL") / len(short) * 100, 2) if short else 0
    p_cont_long = round(sum(1 for s in long_ if s["outcome"] == "CONTINUATION") / len(long_) * 100, 2) if long_ else 0

    # Balance effect
    equal_bal = [s for s in valid if s["balance"] == "EQUAL"]
    unequal_bal = [s for s in valid if s["balance"] != "EQUAL"]
    p_cont_equal = round(sum(1 for s in equal_bal if s["outcome"] == "CONTINUATION") / len(equal_bal) * 100, 2) if equal_bal else 0
    p_rev_unequal = round(sum(1 for s in unequal_bal if s["outcome"] == "REVERSAL") / len(unequal_bal) * 100, 2) if unequal_bal else 0

    # RSI effect
    with_rsi = [s for s in valid if s["has_rsi_sequence"]]
    without_rsi = [s for s in valid if not s["has_rsi_sequence"]]
    p_cont_with_rsi = round(sum(1 for s in with_rsi if s["outcome"] == "CONTINUATION") / len(with_rsi) * 100, 2) if with_rsi else 0
    p_rev_without_rsi = round(sum(1 for s in without_rsi if s["outcome"] == "REVERSAL") / len(without_rsi) * 100, 2) if without_rsi else 0

    return {
        "total_segments": total,
        "pct_reversal": pct_reversal,
        "pct_continuation": pct_continuation,
        "p_rev_short": p_rev_short,
        "p_cont_long": p_cont_long,
        "p_cont_equal_balance": p_cont_equal,
        "p_rev_unequal_balance": p_rev_unequal,
        "p_cont_with_rsi_seq": p_cont_with_rsi,
        "p_rev_without_rsi_seq": p_rev_without_rsi,
        "n_short": len(short),
        "n_long": len(long_),
        "n_equal_bal": len(equal_bal),
        "n_unequal_bal": len(unequal_bal),
        "n_with_rsi": len(with_rsi),
        "n_without_rsi": len(without_rsi),
    }


# ============================================================================
# STEP 6 — BLOCK STABILITY (TIME)
# ============================================================================

def block_stability(redist_segments: list[dict], n_blocks: int = 4) -> list[dict]:
    """Split segments into time blocks and compute metrics per block."""
    valid = [s for s in redist_segments if s["outcome"] in ("REVERSAL", "CONTINUATION")]
    if not valid:
        return []

    valid.sort(key=lambda s: s["start_ts"])
    block_size = len(valid) // n_blocks
    blocks: list[dict] = []

    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else len(valid)
        block_segs = valid[start:end]
        metrics = compute_metrics(block_segs)
        ts_start = datetime.fromtimestamp(block_segs[0]["start_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        ts_end = datetime.fromtimestamp(block_segs[-1]["end_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        metrics["block"] = b + 1
        metrics["period"] = f"{ts_start} → {ts_end}"
        blocks.append(metrics)

    return blocks


# ============================================================================
# STEP 7 — DRIFT TEST
# ============================================================================

def drift_test(blocks: list[dict]) -> dict:
    """Compare first block vs last block."""
    if len(blocks) < 2:
        return {"error": "need at least 2 blocks"}

    first = blocks[0]
    last = blocks[-1]
    keys = ["pct_reversal", "p_rev_short", "p_cont_long", "p_cont_equal_balance", "p_rev_unequal_balance"]
    drift: dict[str, float] = {}
    for k in keys:
        v1 = first.get(k, 0)
        v2 = last.get(k, 0)
        drift[k] = round(v2 - v1, 2)

    return {"first_block": first, "last_block": last, "drift": drift}


# ============================================================================
# PIPELINE: run full analysis for one asset + one fractal
# ============================================================================

def run_pipeline(df: pd.DataFrame, fractal: int = 1) -> dict:
    """Run complete pipeline for one dataset and one fractal definition."""
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], period=14)

    if fractal == 1:
        swings = detect_swings_fractal1(df)
    else:
        swings = detect_swings_fractal2(df)

    swings = label_micro_states(swings)
    swings = classify_structural_states(swings)
    segments = build_state_path(swings)

    redist = label_redistribution_outcomes(segments)
    redist = compute_behavior_features(redist)

    metrics = compute_metrics(redist)
    blocks = block_stability(redist)
    drift = drift_test(blocks)

    return {
        "total_swings": len(swings),
        "total_state_segments": len(segments),
        "total_redist_segments": len(redist),
        "metrics": metrics,
        "blocks": blocks,
        "drift": drift,
    }


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def _short(s: str) -> str:
    return s.replace("CONTINUATION_", "CONT_").replace("REDISTRIBUTION", "REDIST")


def format_report(all_results: dict, gap_infos: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  REDISTRIBUTION BEHAVIOR VALIDATION REPORT")
    lines.append("=" * 80)

    assets = list(all_results.keys())

    # Data summary
    lines.append("\n--- DATA SUMMARY ---")
    for asset in assets:
        gi = gap_infos[asset]
        lines.append(f"  {asset}: {gi['total_candles']:,} candles, "
                     f"{gi['missing_segments']} gaps (max {gi['largest_gap_minutes']} min)")

    # Per-asset, per-fractal results
    for asset in assets:
        for frac in [1, 2]:
            key = f"fractal{frac}"
            r = all_results[asset][key]
            m = r["metrics"]
            lines.append(f"\n{'=' * 80}")
            lines.append(f"  {asset} — Fractal {frac}")
            lines.append(f"{'=' * 80}")

            lines.append(f"\n  1. GLOBAL")
            lines.append(f"     Total swings: {r['total_swings']}")
            lines.append(f"     Total state segments: {r['total_state_segments']}")
            lines.append(f"     Total redistribution segments: {m.get('total_segments', 0)}")
            lines.append(f"     % Reversal: {m.get('pct_reversal', 0):.2f}%")
            lines.append(f"     % Continuation: {m.get('pct_continuation', 0):.2f}%")

            lines.append(f"\n  2. LENGTH EFFECT")
            lines.append(f"     P(reversal | swings<=2):    {m.get('p_rev_short', 0):.2f}%  (n={m.get('n_short', 0)})")
            lines.append(f"     P(continuation | swings>=3): {m.get('p_cont_long', 0):.2f}%  (n={m.get('n_long', 0)})")

            lines.append(f"\n  3. BALANCE EFFECT")
            lines.append(f"     P(continuation | balance==EQUAL): {m.get('p_cont_equal_balance', 0):.2f}%  (n={m.get('n_equal_bal', 0)})")
            lines.append(f"     P(reversal | balance!=EQUAL):     {m.get('p_rev_unequal_balance', 0):.2f}%  (n={m.get('n_unequal_bal', 0)})")

            lines.append(f"\n  4. RSI EFFECT")
            lines.append(f"     P(continuation | RSI sequence): {m.get('p_cont_with_rsi_seq', 0):.2f}%  (n={m.get('n_with_rsi', 0)})")
            lines.append(f"     P(reversal | no RSI sequence):  {m.get('p_rev_without_rsi_seq', 0):.2f}%  (n={m.get('n_without_rsi', 0)})")

            # Per-block
            blocks = r.get("blocks", [])
            if blocks:
                lines.append(f"\n  5. PER-BLOCK STABILITY")
                headers = ["Block", "Period", "Segments", "%Rev", "P(rev|short)", "P(cont|long)", "P(cont|eq_bal)"]
                rows = []
                for bl in blocks:
                    rows.append([
                        bl["block"], bl["period"], bl.get("total_segments", 0),
                        f"{bl.get('pct_reversal', 0):.1f}", f"{bl.get('p_rev_short', 0):.1f}",
                        f"{bl.get('p_cont_long', 0):.1f}", f"{bl.get('p_cont_equal_balance', 0):.1f}",
                    ])
                lines.append(tabulate(rows, headers=headers, tablefmt="simple"))

                # Variance
                rev_vals = [bl.get("pct_reversal", 0) for bl in blocks]
                lines.append(f"\n     Variance of %reversal across blocks: {np.var(rev_vals):.2f}")

            # Drift
            drift = r.get("drift", {})
            if "drift" in drift:
                lines.append(f"\n  6. DRIFT (first block vs last block)")
                for k, v in drift["drift"].items():
                    lines.append(f"     {k}: {v:+.2f}")

    # STEP 8 — Swing robustness
    lines.append(f"\n{'=' * 80}")
    lines.append("  SWING ROBUSTNESS: Fractal 1 vs Fractal 2")
    lines.append(f"{'=' * 80}")
    headers = ["Asset", "Metric", "Fractal 1", "Fractal 2", "Diff"]
    rows = []
    for asset in assets:
        m1 = all_results[asset]["fractal1"]["metrics"]
        m2 = all_results[asset]["fractal2"]["metrics"]
        for metric_key, label in [
            ("total_segments", "Redist count"),
            ("pct_reversal", "% Reversal"),
            ("p_rev_short", "P(rev|short)"),
            ("p_cont_long", "P(cont|long)"),
        ]:
            v1 = m1.get(metric_key, 0)
            v2 = m2.get(metric_key, 0)
            diff = v2 - v1 if isinstance(v1, (int, float)) else "N/A"
            rows.append([asset, label, v1, v2, f"{diff:+.2f}" if isinstance(diff, (int, float)) else diff])
    lines.append(tabulate(rows, headers=headers, tablefmt="simple"))

    # STEP 9 — Cross-asset
    lines.append(f"\n{'=' * 80}")
    lines.append("  CROSS-ASSET VALIDATION (Fractal 1)")
    lines.append(f"{'=' * 80}")
    headers = ["Metric"] + assets
    metrics_to_show = [
        ("pct_reversal", "% Reversal"),
        ("pct_continuation", "% Continuation"),
        ("p_rev_short", "P(rev|short)"),
        ("p_cont_long", "P(cont|long)"),
        ("p_cont_equal_balance", "P(cont|eq_bal)"),
        ("p_rev_unequal_balance", "P(rev|uneq_bal)"),
    ]
    rows = []
    for mk, label in metrics_to_show:
        row = [label]
        for asset in assets:
            val = all_results[asset]["fractal1"]["metrics"].get(mk, 0)
            row.append(f"{val:.2f}")
        rows.append(row)
    lines.append(tabulate(rows, headers=headers, tablefmt="simple"))

    # FINAL CONCLUSION placeholder
    lines.append(f"\n{'=' * 80}")
    lines.append("  FINAL CONCLUSION")
    lines.append(f"{'=' * 80}")

    # Auto-generate a simple conclusion
    # Check stability: variance of %reversal across blocks
    conclusions: list[str] = []
    for asset in assets:
        blocks = all_results[asset]["fractal1"].get("blocks", [])
        if blocks:
            rev_vals = [bl.get("pct_reversal", 0) for bl in blocks]
            var = np.var(rev_vals)
            stability = "STABLE" if var < 50 else "MODERATE" if var < 100 else "UNSTABLE"
            conclusions.append(f"  {asset} (Fractal 1): block variance={var:.2f} → {stability}")

    # Check swing robustness
    for asset in assets:
        m1 = all_results[asset]["fractal1"]["metrics"]
        m2 = all_results[asset]["fractal2"]["metrics"]
        diff = abs(m1.get("pct_reversal", 0) - m2.get("pct_reversal", 0))
        robust = "ROBUST" if diff < 5 else "MODERATE" if diff < 10 else "SENSITIVE"
        conclusions.append(f"  {asset} swing definition: diff={diff:.2f}% → {robust}")

    # Check strongest feature
    for asset in assets:
        m = all_results[asset]["fractal1"]["metrics"]
        features = {
            "length": abs(m.get("p_rev_short", 50) - 50) + abs(m.get("p_cont_long", 50) - 50),
            "balance": abs(m.get("p_cont_equal_balance", 50) - 50) + abs(m.get("p_rev_unequal_balance", 50) - 50),
            "RSI": abs(m.get("p_cont_with_rsi_seq", 50) - 50) + abs(m.get("p_rev_without_rsi_seq", 50) - 50),
        }
        strongest = max(features, key=lambda k: features[k])
        conclusions.append(f"  {asset} strongest feature: {strongest} (deviation={features[strongest]:.1f})")

    lines.append("\n  Behavior stability:")
    lines.extend(conclusions)

    lines.append(f"\n{'=' * 80}")
    lines.append("  END OF VALIDATION REPORT")
    lines.append(f"{'=' * 80}")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

PAIRS = ["DOGE/USDT", "BTC/USDT", "SOL/USDT"]


def main():
    all_results: dict = {}
    gap_infos: dict = {}

    for pair in PAIRS:
        asset_name = pair.split("/")[0]
        print(f"\n{'='*60}")
        print(f"Fetching {pair} 15m data (max 4 years)...")
        print(f"{'='*60}")

        df = fetch_ohlcv(pair, "15m", years=4)
        print(f"  → {len(df):,} candles fetched")

        gap_infos[asset_name] = check_missing_candles(df)
        print(f"  → Missing segments: {gap_infos[asset_name]['missing_segments']}")

        all_results[asset_name] = {}

        for frac in [1, 2]:
            print(f"\n  Running pipeline with Fractal {frac}...")
            result = run_pipeline(df, fractal=frac)
            all_results[asset_name][f"fractal{frac}"] = result
            m = result["metrics"]
            print(f"    Swings: {result['total_swings']}, "
                  f"Segments: {result['total_state_segments']}, "
                  f"Redist: {m.get('total_segments', 0)}, "
                  f"Rev: {m.get('pct_reversal', 0):.1f}%, "
                  f"Cont: {m.get('pct_continuation', 0):.1f}%")

    print("\n\nGenerating report...\n")
    report = format_report(all_results, gap_infos)
    print(report)

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    json_path = os.path.join(output_dir, "validation_raw.json")
    # Strip swing-level data for JSON (too large)
    export = {}
    for asset, fractals in all_results.items():
        export[asset] = {}
        for fkey, fdata in fractals.items():
            export[asset][fkey] = {
                "total_swings": fdata["total_swings"],
                "total_state_segments": fdata["total_state_segments"],
                "metrics": fdata["metrics"],
                "blocks": fdata["blocks"],
                "drift": fdata["drift"],
            }
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"Raw data saved to: {json_path}")


if __name__ == "__main__":
    main()
