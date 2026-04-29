# test-n8n

## Market Structure Analysis Tools

Tools for discovering recurring paths of market behavior (Continuation ↔ Redistribution) using swing structure and relational RSI.

### Scripts

| Script | Description |
|--------|-------------|
| `doge_structure_analysis.py` | DOGE/USDT 15m — state path analysis (swing detection, micro states, structural classification, transition matrix, RSI relational behavior) |
| `validate_redistribution.py` | Multi-asset validation of redistribution behavior stability across time, swing definitions, and assets (DOGE, BTC, SOL) |

### Quick Start

```bash
pip install -r requirements.txt

# Run single-asset structure analysis (DOGE/USDT 15m, 12 months)
python doge_structure_analysis.py

# Run full validation (3 assets, 2 fractal defs, 4-year history)
python validate_redistribution.py
```

### Methodology

See `VALIDATION_SPEC.md` for the full validation specification.

**Core concepts:**
- **Swing detection**: Fractal-based (1-bar and 2-bar lookback/lookahead)
- **Micro states**: HH, LH, HL, LL (relative to previous swing of same type)
- **Structural states**: CONTINUATION_UP, CONTINUATION_DOWN, REDISTRIBUTION
- **RSI**: Wilder (14), relational only — no thresholds
- **Output**: State counts, transition matrix, recurring paths, reversal analysis, drift test

### Output

Reports are saved to the `data/` directory:
- `data/structure_analysis_report.txt` — single-asset report
- `data/structure_analysis_raw.json` — raw data (JSON)
- `data/validation_report.txt` — multi-asset validation report
- `data/validation_raw.json` — raw validation data (JSON)
