# Redistribution Behavior Validation Spec

## GOAL
Validate stability of redistribution behavior across:
- Long time (2–4 years)
- Multiple assets
- Different swing definitions

## Focus
- Behavior only (NOT prediction)
- 2 behaviors:
  1. **REVERSAL** (đảo chiều)
  2. **CONTINUATION** (tiếp diễn)

---

## DATA

| Parameter | Value |
|-----------|-------|
| Pairs | DOGE/USDT, BTC/USDT, SOL/USDT (mid-cap) |
| Timeframe | 15m |
| Period | 2–4 years (max available) |
| Requirements | No missing candles, no interpolation |
| Indicator | RSI (14, Wilder) — relational only |

---

## SWING DEFINITIONS (Test Robustness)

Run ALL analysis with 2 versions:

### A) Fractal 1
```
low[i] < low[i-1] AND low[i] < low[i+1]
high[i] > high[i-1] AND high[i] > high[i+1]
```

### B) Fractal 2
```
low[i] < low[i-1] AND low[i] < low[i-2] AND low[i] < low[i+1] AND low[i] < low[i+2]
high[i] > high[i-1] AND high[i] > high[i-2] AND high[i] > high[i+1] AND high[i] > high[i+2]
```

---

## STATE DEFINITION (FIXED)

| State | Condition |
|-------|-----------|
| CONTINUATION_UP | HH + HL |
| CONTINUATION_DOWN | LH + LL |
| REDISTRIBUTION | First structural break + confirmation swing |

---

## STEPS

### Step 1 — Extract Redistribution
- Extract all redistribution segments per dataset
- Store full swing list per segment

### Step 2 — Label Outcome
- Next state opposite → **REVERSAL**
- Next state same → **CONTINUATION**

### Step 3 — Core Behavior Features
Per segment:
1. `num_swings`
2. `structure_sequence` (HH/HL/LH/LL)
3. `balance`: (HH+HL) vs (LH+LL) → UP / DOWN / EQUAL
4. RSI relational: RSI_high sequence (>,<,=) and RSI_low sequence (>,<,=)

### Step 4 — Behavior Rules (DO NOT CHANGE)

**REVERSAL behavior:**
- num_swings <= 2
- structure in: LH→LL, HL→HH, LL→LH, HH→HL
- balance != EQUAL
- RSI minimal / no sequence

**CONTINUATION behavior:**
- num_swings >= 3
- mixed structure (>=3 swings)
- balance == EQUAL
- RSI has sequence (>=2 comparisons)

### Step 5 — Core Metrics
1. Distribution: % REVERSAL vs % CONTINUATION
2. Length effect: P(reversal | swings<=2), P(continuation | swings>=3)
3. Balance effect: P(continuation | balance==EQUAL), P(reversal | balance!=EQUAL)
4. RSI effect: segments with RSI sequence vs none

### Step 6 — Block Stability (Time)
- Split dataset into 4 equal time blocks
- Compute all metrics per block
- Compare variance between blocks, check drift

### Step 7 — Drift Test (QUAN TRỌNG)
- Compare first block vs last block
- Check: reversal %, length effect, balance effect
- Output difference values

### Step 8 — Swing Robustness
- Compare Fractal 1 vs Fractal 2
- Check: redistribution count, reversal %, behavior metrics
- Large difference → unstable definition

### Step 9 — Cross-Asset Validation
- Compare DOGE vs BTC vs mid-cap
- Check distribution similarity, behavior consistency

---

## OUTPUT

1. **GLOBAL**: total segments, % reversal vs continuation
2. **PER BLOCK**: stats per block, variance
3. **DRIFT**: early vs late comparison
4. **SWING TEST**: fractal 1 vs 2 comparison
5. **CROSS-ASSET**: DOGE vs BTC vs mid-cap
6. **FINAL CONCLUSION**: Is behavior stable? Strongest features (length/balance/RSI)?

---

## IMPORTANT
- DO NOT optimize rules
- DO NOT add thresholds
- DO NOT change behavior definition
- DO NOT filter data
- **ONLY validate behavior**
