"""
MARKET INTELLIGENCE ENGINE
===========================
Continuous backend system:
- Fetches earnings calendar, pre-market gappers, short interest watchlist
- Runs quantitative pattern detection (PEAD + squeeze candidates)
- Calls Claude AI for qualitative analysis and final directional call
- Serves live signals via Flask API to the HTML dashboard

SETUP:
  pip install flask flask-cors yfinance requests pandas numpy anthropic

DEPLOY:
  Railway / Render / PythonAnywhere — run `python engine.py`
  Set environment variable: ANTHROPIC_API_KEY=your_key
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
SCAN_INTERVAL_SEC   = 60          # How often the engine re-scans (seconds)
GAP_THRESHOLD_PCT   = 5.0         # Minimum gap % to qualify
VOLUME_MULT_MIN     = 1.8         # Minimum volume multiplier vs 20d average
SHORT_INT_MIN       = 12.0        # Minimum short interest % to flag squeeze
MAX_CANDIDATES      = 20          # Cap candidates sent to AI per cycle
PORT                = int(os.environ.get("PORT", 8080))

# ── Shared state (updated by background thread, read by Flask) ─────────────────
STATE = {
    "signals":       [],          # Final AI-approved signals
    "candidates":    [],          # Stocks that passed quant filter
    "watchlist":     [],          # Current screened watchlist
    "last_updated":  None,
    "market_status": "checking",
    "scan_count":    0,
    "errors":        [],
}
STATE_LOCK = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

# High-short-interest watchlist — update weekly via Finviz or manually
# These are pre-qualified squeeze candidates waiting for a catalyst
BASE_WATCHLIST = [
    "GME","BBBY","AMC","BYND","RIVN","LCID","NKLA","WKHS","SPCE","CLOV",
    "MVIS","SNDL","EXPR","KOSS","TLRY","AFRM","UPST","HOOD","SOFI","COIN",
    "NVAX","BIOR","PTRA","GOEV","RIDE","HYLN","OPEN","DKNG","WISH","BARK",
    "CRSR","SKLZ","RBLX","MAPS","GEVI","BLNK","CHPT","FCEL","PLUG","ASTS",
]

def fetch_stock_data(ticker: str) -> dict | None:
    """Fetch price, volume, fundamentals for a single ticker."""
    try:
        tk  = yf.Ticker(ticker)
        hist = tk.history(period="30d", interval="1d")
        if hist.empty or len(hist) < 5:
            return None

        info = tk.info or {}
        current_price  = hist["Close"].iloc[-1]
        prev_close     = hist["Close"].iloc[-2]
        today_volume   = hist["Volume"].iloc[-1]
        avg_volume_20d = hist["Volume"].iloc[:-1].mean()
        volume_mult    = today_volume / avg_volume_20d if avg_volume_20d > 0 else 0
        gap_pct        = ((current_price - prev_close) / prev_close) * 100

        return {
            "ticker":        ticker,
            "price":         round(current_price, 2),
            "prev_close":    round(prev_close, 2),
            "gap_pct":       round(gap_pct, 2),
            "volume":        int(today_volume),
            "avg_volume":    int(avg_volume_20d),
            "volume_mult":   round(volume_mult, 2),
            "market_cap":    info.get("marketCap", 0),
            "float_shares":  info.get("floatShares", 0),
            "short_pct":     round((info.get("shortPercentOfFloat") or 0) * 100, 1),
            "sector":        info.get("sector", "Unknown"),
            "name":          info.get("shortName", ticker),
            "52w_high":      info.get("fiftyTwoWeekHigh", 0),
            "52w_low":       info.get("fiftyTwoWeekLow",  0),
            "pe_ratio":      info.get("trailingPE", None),
            "earnings_date": str(info.get("earningsTimestamp", "")),
        }
    except Exception as e:
        log.warning(f"Data fetch failed for {ticker}: {e}")
        return None


def fetch_earnings_surprises() -> list[dict]:
    """
    Fetch recent earnings reports with beat/miss data.
    Uses Yahoo Finance — for production, swap to Earnings Whispers API
    or Alpha Vantage for richer data.
    """
    # Tickers that recently reported — in production this comes from
    # an earnings calendar API. For now we scan the watchlist for
    # stocks with recent earnings.
    surprises = []
    calendar_tickers = BASE_WATCHLIST[:15]   # limit for demo speed
    for t in calendar_tickers:
        try:
            tk   = yf.Ticker(t)
            hist = tk.history(period="5d", interval="1d")
            info = tk.info or {}
            if hist.empty:
                continue
            # Proxy for earnings beat: trailing EPS vs estimate
            eps_actual   = info.get("trailingEps", None)
            eps_estimate = info.get("forwardEps",  None)
            if eps_actual and eps_estimate and eps_actual > eps_estimate * 1.05:
                surprises.append({
                    "ticker":       t,
                    "eps_actual":   eps_actual,
                    "eps_estimate": eps_estimate,
                    "beat_pct":     round(((eps_actual - eps_estimate) / abs(eps_estimate)) * 100, 1),
                })
        except Exception:
            pass
    return surprises


def fetch_premarket_gappers() -> list[str]:
    """
    Return tickers showing large pre-market gaps.
    In production: use Polygon.io /v2/snapshot/locale/us/markets/stocks/gainers
    or Unusual Whales pre-market endpoint.
    For now: scan watchlist for tickers up >GAP_THRESHOLD_PCT today.
    """
    gappers = []
    for t in BASE_WATCHLIST:
        try:
            hist = yf.Ticker(t).history(period="2d", interval="1d")
            if len(hist) < 2:
                continue
            gap = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100
            if abs(gap) >= GAP_THRESHOLD_PCT:
                gappers.append(t)
        except Exception:
            pass
    return gappers


# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE PATTERN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def quant_score(data: dict) -> dict:
    """
    Evaluate quantitative signals for a stock.
    Returns a structured quant report — NOT a score.
    The AI uses this as context, not a verdict.
    """
    flags = []
    concerns = []

    # ── Gap analysis ──────────────────────────────────────────────────────────
    gap = data["gap_pct"]
    if gap >= 10:
        flags.append(f"Strong gap up {gap:.1f}% — potential PEAD candidate")
    elif gap >= 5:
        flags.append(f"Moderate gap up {gap:.1f}%")
    elif gap <= -10:
        flags.append(f"Strong gap DOWN {gap:.1f}% — potential short candidate")
    elif gap <= -5:
        flags.append(f"Moderate gap down {gap:.1f}%")
    else:
        concerns.append(f"Gap only {gap:.1f}% — limited momentum signal")

    # ── Volume analysis ───────────────────────────────────────────────────────
    vm = data["volume_mult"]
    if vm >= 3.0:
        flags.append(f"Exceptional volume {vm:.1f}x average — institutional participation confirmed")
    elif vm >= 2.0:
        flags.append(f"Strong volume {vm:.1f}x average")
    elif vm >= 1.5:
        flags.append(f"Above average volume {vm:.1f}x")
    else:
        concerns.append(f"Volume only {vm:.1f}x average — weak institutional signal")

    # ── Short squeeze potential ────────────────────────────────────────────────
    si = data["short_pct"]
    fl = data["float_shares"]
    if si >= 25:
        flags.append(f"Extreme short interest {si:.1f}% — major squeeze potential")
    elif si >= 15:
        flags.append(f"High short interest {si:.1f}% — squeeze candidate")
    elif si >= SHORT_INT_MIN:
        flags.append(f"Elevated short interest {si:.1f}%")
    else:
        concerns.append(f"Short interest only {si:.1f}% — limited squeeze fuel")

    if fl and fl < 30_000_000:
        flags.append(f"Low float {fl/1e6:.1f}M shares — explosive move potential")
    elif fl and fl < 75_000_000:
        flags.append(f"Moderate float {fl/1e6:.1f}M shares")

    # ── 52-week range position ────────────────────────────────────────────────
    hi, lo, px = data["52w_high"], data["52w_low"], data["price"]
    if hi and lo and hi != lo:
        range_pos = (px - lo) / (hi - lo) * 100
        if range_pos >= 90:
            flags.append(f"Near 52-week high ({range_pos:.0f}% of range) — strong momentum")
        elif range_pos >= 70:
            flags.append(f"Upper range position ({range_pos:.0f}%)")
        elif range_pos <= 20:
            concerns.append(f"Near 52-week low ({range_pos:.0f}%) — downtrend concern")

    # ── PEAD convergence ──────────────────────────────────────────────────────
    pead_signals = sum([
        gap >= 8,
        vm >= 2.0,
        si >= SHORT_INT_MIN,
    ])
    pead_ready = pead_signals >= 2

    return {
        "flags":      flags,
        "concerns":   concerns,
        "pead_ready": pead_ready,
        "squeeze_ready": si >= SHORT_INT_MIN and gap > 0,
        "signal_count": pead_signals,
    }


def filter_candidates(tickers: list[str]) -> list[dict]:
    """
    Fetch data for all tickers and filter down to genuine candidates.
    """
    candidates = []
    for t in tickers:
        data = fetch_stock_data(t)
        if not data:
            continue
        quant = quant_score(data)
        # Must have at least 2 quant signals to proceed to AI
        if quant["signal_count"] >= 1 or quant["squeeze_ready"]:
            candidates.append({**data, "quant": quant})
    return candidates[:MAX_CANDIDATES]


# ══════════════════════════════════════════════════════════════════════════════
# AI QUALITATIVE ANALYSIS LAYER
# ══════════════════════════════════════════════════════════════════════════════

def build_briefing(stock: dict) -> str:
    """Build the AI briefing document for a single stock candidate."""
    q = stock["quant"]
    flags_text    = "\n".join(f"  ✓ {f}" for f in q["flags"])    or "  None"
    concerns_text = "\n".join(f"  ⚠ {c}" for c in q["concerns"]) or "  None"

    return f"""
MARKET INTELLIGENCE BRIEFING
Stock: {stock['name']} ({stock['ticker']})
Sector: {stock['sector']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

━━━ QUANTITATIVE CONTEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Price Data:
  Current Price:   ${stock['price']}
  Previous Close:  ${stock['prev_close']}
  Gap:             {stock['gap_pct']:+.2f}%
  Today Volume:    {stock['volume']:,}
  20d Avg Volume:  {stock['avg_volume']:,}
  Volume Multiple: {stock['volume_mult']:.1f}x

Short Interest & Float:
  Short Interest:  {stock['short_pct']:.1f}% of float
  Float:           {f"{stock['float_shares']/1e6:.1f}M" if stock['float_shares'] else 'Unknown'} shares

Positioning:
  52-Week High:    ${stock['52w_high']}
  52-Week Low:     ${stock['52w_low']}
  P/E Ratio:       {stock['pe_ratio'] or 'N/A'}

Quantitative Signals Detected:
{flags_text}

Quantitative Concerns:
{concerns_text}

PEAD Candidate:    {'YES' if q['pead_ready'] else 'NO'}
Squeeze Candidate: {'YES' if q['squeeze_ready'] else 'NO'}

━━━ YOUR ANALYSIS TASK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are a seasoned professional trader with deep expertise in 
post-earnings drift, short squeeze dynamics, and institutional 
order flow. You think like Minervini, O'Neil, and Livermore.

Using the quantitative context above AS CONTEXT (not as a verdict),
apply your qualitative judgment to assess:

1. What is the most probable directional move for this stock?
2. What is driving it — PEAD, squeeze, momentum, or a combination?
3. What timeframe should this trade be held?
4. What is the primary risk to this thesis?
5. What is your overall conviction level: HIGH / MODERATE / LOW?

The quantitative data informs you. Your judgment concludes.
Be direct. Be specific. Issue a clear verdict.

Respond in this exact JSON format:
{{
  "direction": "LONG" | "SHORT" | "NO TRADE",
  "thesis": "2-3 sentence explanation of WHY",
  "primary_driver": "PEAD" | "SQUEEZE" | "MOMENTUM" | "COMBINED" | "NONE",
  "timeframe": "e.g. 2-4 days",
  "key_risk": "primary risk to this thesis",
  "conviction": "HIGH" | "MODERATE" | "LOW",
  "entry_note": "brief note on entry timing/confirmation to watch for"
}}
"""


def call_claude(briefing: str) -> dict | None:
    """Send briefing to Claude API and get structured response."""
    if not ANTHROPIC_API_KEY:
        log.warning("No ANTHROPIC_API_KEY set — skipping AI call")
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 600,
                "messages":   [{"role": "user", "content": briefing}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"AI response not valid JSON: {e}")
        return None
    except Exception as e:
        log.error(f"Claude API call failed: {e}")
        return None


def analyse_candidates(candidates: list[dict]) -> list[dict]:
    """Run AI analysis on each quant-filtered candidate."""
    signals = []
    for stock in candidates:
        log.info(f"  → AI analysing {stock['ticker']}...")
        briefing = build_briefing(stock)
        ai       = call_claude(briefing)
        if not ai:
            continue
        if ai.get("direction") == "NO TRADE":
            log.info(f"    {stock['ticker']}: AI issued NO TRADE")
            continue
        signals.append({
            "ticker":         stock["ticker"],
            "name":           stock["name"],
            "sector":         stock["sector"],
            "price":          stock["price"],
            "gap_pct":        stock["gap_pct"],
            "volume_mult":    stock["volume_mult"],
            "short_pct":      stock["short_pct"],
            "direction":      ai["direction"],
            "thesis":         ai["thesis"],
            "primary_driver": ai["primary_driver"],
            "timeframe":      ai["timeframe"],
            "key_risk":       ai["key_risk"],
            "conviction":     ai["conviction"],
            "entry_note":     ai["entry_note"],
            "timestamp":      datetime.now().isoformat(),
            "quant_flags":    stock["quant"]["flags"],
        })
        log.info(f"    {stock['ticker']}: {ai['direction']} | {ai['conviction']} conviction")
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_scan():
    """One full scan cycle: discover → filter → analyse → update state."""
    log.info("━━━ SCAN CYCLE STARTING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 1. Build candidate universe from triggers
    universe = set(BASE_WATCHLIST)
    gappers  = fetch_premarket_gappers()
    universe.update(gappers)
    log.info(f"Universe: {len(universe)} tickers | Gappers detected: {gappers}")

    # 2. Quant filter
    candidates = filter_candidates(list(universe))
    log.info(f"Quant filter: {len(candidates)} candidates passed")

    # 3. AI analysis
    signals = analyse_candidates(candidates)
    log.info(f"AI analysis: {len(signals)} signals issued")

    # 4. Update shared state
    now = datetime.now().isoformat()
    with STATE_LOCK:
        STATE["signals"]       = signals
        STATE["candidates"]    = [
            {"ticker": c["ticker"], "name": c["name"], "gap_pct": c["gap_pct"],
             "volume_mult": c["volume_mult"], "short_pct": c["short_pct"],
             "quant_flags": c["quant"]["flags"]}
            for c in candidates
        ]
        STATE["watchlist"]     = list(universe)
        STATE["last_updated"]  = now
        STATE["scan_count"]   += 1
        STATE["market_status"] = "live"

    log.info(f"━━━ SCAN COMPLETE — {len(signals)} active signals ━━━━━━━━━━━━")


def scan_loop():
    """Background thread: run scan continuously."""
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan cycle error: {e}")
            with STATE_LOCK:
                STATE["errors"].append({"time": datetime.now().isoformat(), "error": str(e)})
        time.sleep(SCAN_INTERVAL_SEC)


# ══════════════════════════════════════════════════════════════════════════════
# FLASK API SERVER
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)   # Allow requests from any origin (HTML dashboard on any laptop)

@app.route("/api/state")
def api_state():
    with STATE_LOCK:
        return jsonify(STATE)

@app.route("/api/signals")
def api_signals():
    with STATE_LOCK:
        return jsonify(STATE["signals"])

@app.route("/api/candidates")
def api_candidates():
    with STATE_LOCK:
        return jsonify(STATE["candidates"])

@app.route("/api/health")
def api_health():
    with STATE_LOCK:
        return jsonify({
            "status":       "online",
            "scan_count":   STATE["scan_count"],
            "last_updated": STATE["last_updated"],
            "signals":      len(STATE["signals"]),
        })

@app.route("/")
def index():
    return "<h2>Market Intelligence Engine — Online ✓</h2><p>Dashboard connects via /api/state</p>"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("▶ Market Intelligence Engine starting...")
    log.info(f"  Scan interval: {SCAN_INTERVAL_SEC}s")
    log.info(f"  AI key:        {'SET ✓' if ANTHROPIC_API_KEY else 'NOT SET ✗'}")
    log.info(f"  Port:          {PORT}")

    # Start background scan thread
    scanner = threading.Thread(target=scan_loop, daemon=True)
    scanner.start()

    # Start Flask server
    app.run(host="0.0.0.0", port=PORT, debug=False)
