"""
MARKET INTELLIGENCE ENGINE v2
===============================
Additions in this version:
- ATR-14 calculation (Average True Range)
- VWAP approximation from daily data
- ATR-based stop loss (1.5x ATR for normal, 2.0x for volatile/squeeze)
- Structural stop (gap open level, prior high/low)
- Two take profit targets (2:1 R/R and measured move)
- Entry zone calculation (VWAP hold, opening range, pullback zone)
- All trade levels passed to AI briefing and returned in signal

SETUP:
  pip install flask flask-cors yfinance requests pandas numpy

DEPLOY:
  Railway / Render / PythonAnywhere — run `python engine.py`
  Set environment variable: ANTHROPIC_API_KEY=your_key
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SCAN_INTERVAL_SEC = 60
GAP_THRESHOLD_PCT = 5.0
VOLUME_MULT_MIN   = 1.8
SHORT_INT_MIN     = 12.0
MAX_CANDIDATES    = 20
PORT              = int(os.environ.get("PORT", 8080))

STATE = {
    "signals":      [],
    "candidates":   [],
    "watchlist":    [],
    "last_updated": None,
    "scan_count":   0,
    "errors":       [],
}
STATE_LOCK = threading.Lock()

BASE_WATCHLIST = [
    "GME","AMC","BYND","RIVN","LCID","SPCE","CLOV",
    "MVIS","SNDL","TLRY","AFRM","UPST","HOOD","SOFI","COIN",
    "NVAX","OPEN","DKNG","BARK","CRSR","SKLZ","RBLX",
    "BLNK","CHPT","FCEL","PLUG","ASTS","JOBY","NKLA","WKHS",
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA + ATR LAYER
# ══════════════════════════════════════════════════════════════════════════════

def calc_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range over last N periods."""
    try:
        high  = hist["High"]
        low   = hist["Low"]
        close = hist["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low  - close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0


def calc_vwap(hist: pd.DataFrame) -> float:
    """Approximate VWAP from today's OHLCV (daily bar)."""
    try:
        row    = hist.iloc[-1]
        typical = (row["High"] + row["Low"] + row["Close"]) / 3
        return round(float(typical), 2)
    except Exception:
        return 0.0


def calc_trade_levels(data: dict, hist: pd.DataFrame, direction: str, driver: str) -> dict:
    """
    Calculate ATR-based stop loss, two take profit targets, and entry zone.
    Returns a dict of all trade management levels.
    """
    price    = data["price"]
    prev_close = data["prev_close"]
    gap_open = prev_close * (1 + data["gap_pct"] / 100)   # approximate gap open
    atr      = data.get("atr", 0)
    vwap     = data.get("vwap", price)

    if atr <= 0:
        atr = price * 0.05   # fallback: 5% of price if ATR unavailable

    # ATR multiplier — wider for squeeze/volatile setups
    volatile_drivers = {"SQUEEZE", "COMBINED"}
    atr_mult_stop = 2.0 if driver in volatile_drivers else 1.5
    atr_mult_trail = 1.0

    # ── Stop Loss ──────────────────────────────────────────────────────────────
    if direction == "LONG":
        # ATR stop
        atr_stop = round(price - (atr * atr_mult_stop), 2)
        # Structural stop: just below gap open (gap fill = thesis dead)
        structural_stop = round(gap_open * 0.99, 2)
        # Use the higher of the two (tighter but still outside noise)
        stop_loss = max(atr_stop, structural_stop)
        stop_loss = round(stop_loss, 2)
    else:  # SHORT
        atr_stop = round(price + (atr * atr_mult_stop), 2)
        structural_stop = round(gap_open * 1.01, 2)
        stop_loss = min(atr_stop, structural_stop)
        stop_loss = round(stop_loss, 2)

    # ── Risk per share ─────────────────────────────────────────────────────────
    risk = abs(price - stop_loss)
    if risk < 0.01:
        risk = atr * atr_mult_stop

    # ── Take Profit 1: 2:1 Risk/Reward ────────────────────────────────────────
    if direction == "LONG":
        tp1 = round(price + (risk * 2.0), 2)
    else:
        tp1 = round(price - (risk * 2.0), 2)

    # ── Take Profit 2: Measured Move (gap size projected forward) ─────────────
    gap_size = abs(price - prev_close)
    if direction == "LONG":
        # Project gap size from current price
        tp2 = round(price + gap_size, 2)
        # Also check 52w high as a natural ceiling
        w52_high = data.get("52w_high", 0)
        if w52_high and w52_high > price and w52_high < tp2 * 1.5:
            tp2 = round(w52_high * 0.99, 2)   # just below 52w high
    else:
        tp2 = round(price - gap_size, 2)
        w52_low = data.get("52w_low", 0)
        if w52_low and w52_low < price and w52_low > tp2 * 0.5:
            tp2 = round(w52_low * 1.01, 2)

    # ── Entry Zone ─────────────────────────────────────────────────────────────
    if direction == "LONG":
        # Ideal entry: pullback toward VWAP or gap open, not chasing the spike
        entry_ideal  = round(min(vwap, gap_open) * 1.005, 2)   # just above VWAP/gap open
        entry_max    = round(price * 1.02, 2)                    # do not pay more than 2% above current
        entry_note_computed = f"Enter between ${entry_ideal} and ${entry_max} — wait for pullback to VWAP (${vwap}) or gap open (${round(gap_open,2)}) and first green candle confirmation"
    else:
        entry_ideal  = round(max(vwap, gap_open) * 0.995, 2)
        entry_max    = round(price * 0.98, 2)
        entry_note_computed = f"Enter between ${entry_max} and ${entry_ideal} — wait for dead-cat bounce toward VWAP (${vwap}) to fade, first red candle confirmation"

    # ── Trailing stop for momentum portion ────────────────────────────────────
    trailing_stop_distance = round(atr * atr_mult_trail, 2)

    # ── Risk/Reward ratio actually achieved ───────────────────────────────────
    tp1_rr = round(abs(tp1 - price) / risk, 2) if risk > 0 else 0
    tp2_rr = round(abs(tp2 - price) / risk, 2) if risk > 0 else 0

    return {
        "atr":                    round(atr, 3),
        "vwap":                   round(vwap, 2),
        "gap_open":               round(gap_open, 2),
        "stop_loss":              stop_loss,
        "stop_type":              f"ATR×{atr_mult_stop} + structural",
        "risk_per_share":         round(risk, 2),
        "tp1":                    tp1,
        "tp1_rr":                 tp1_rr,
        "tp1_action":             "Sell 50% of position — move stop to breakeven",
        "tp2":                    tp2,
        "tp2_rr":                 tp2_rr,
        "tp2_action":             "Sell 25% — trail final 25% at ATR×1.0",
        "trailing_stop_distance": trailing_stop_distance,
        "entry_ideal":            entry_ideal,
        "entry_max":              entry_max,
        "entry_note_computed":    entry_note_computed,
        "atr_mult_used":          atr_mult_stop,
    }


def fetch_stock_data(ticker: str) -> dict | None:
    """Fetch price, volume, fundamentals, ATR, VWAP for a single ticker."""
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="30d", interval="1d")
        if hist.empty or len(hist) < 5:
            return None

        info           = tk.info or {}
        current_price  = hist["Close"].iloc[-1]
        prev_close     = hist["Close"].iloc[-2]
        today_volume   = hist["Volume"].iloc[-1]
        avg_volume_20d = hist["Volume"].iloc[:-1].mean()
        volume_mult    = today_volume / avg_volume_20d if avg_volume_20d > 0 else 0
        gap_pct        = ((current_price - prev_close) / prev_close) * 100
        atr            = calc_atr(hist, period=14)
        vwap           = calc_vwap(hist)

        return {
            "ticker":       ticker,
            "price":        round(current_price, 2),
            "prev_close":   round(prev_close, 2),
            "gap_pct":      round(gap_pct, 2),
            "volume":       int(today_volume),
            "avg_volume":   int(avg_volume_20d),
            "volume_mult":  round(volume_mult, 2),
            "market_cap":   info.get("marketCap", 0),
            "float_shares": info.get("floatShares", 0),
            "short_pct":    round((info.get("shortPercentOfFloat") or 0) * 100, 1),
            "sector":       info.get("sector", "Unknown"),
            "name":         info.get("shortName", ticker),
            "52w_high":     info.get("fiftyTwoWeekHigh", 0),
            "52w_low":      info.get("fiftyTwoWeekLow", 0),
            "pe_ratio":     info.get("trailingPE", None),
            "atr":          round(atr, 3),
            "vwap":         round(vwap, 2),
            "atr_pct":      round((atr / current_price) * 100, 2) if current_price > 0 else 0,
        }
    except Exception as e:
        log.warning(f"Data fetch failed for {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE PATTERN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def quant_score(data: dict) -> dict:
    flags, concerns = [], []
    gap = data["gap_pct"]
    vm  = data["volume_mult"]
    si  = data["short_pct"]
    fl  = data["float_shares"]
    atr_pct = data.get("atr_pct", 0)

    if gap >= 10:   flags.append(f"Strong gap up {gap:.1f}% — PEAD candidate")
    elif gap >= 5:  flags.append(f"Moderate gap up {gap:.1f}%")
    elif gap <= -10:flags.append(f"Strong gap DOWN {gap:.1f}% — short candidate")
    elif gap <= -5: flags.append(f"Moderate gap down {gap:.1f}%")
    else: concerns.append(f"Gap only {gap:.1f}% — limited momentum")

    if vm >= 3.0:   flags.append(f"Exceptional volume {vm:.1f}x — institutional confirmed")
    elif vm >= 2.0: flags.append(f"Strong volume {vm:.1f}x average")
    elif vm >= 1.5: flags.append(f"Above average volume {vm:.1f}x")
    else: concerns.append(f"Volume only {vm:.1f}x — weak institutional signal")

    if si >= 25:            flags.append(f"Extreme short interest {si:.1f}% — major squeeze potential")
    elif si >= 15:          flags.append(f"High short interest {si:.1f}% — squeeze candidate")
    elif si >= SHORT_INT_MIN: flags.append(f"Elevated short interest {si:.1f}%")
    else: concerns.append(f"Short interest {si:.1f}% — limited squeeze fuel")

    if fl and fl < 30_000_000:  flags.append(f"Low float {fl/1e6:.1f}M — explosive potential")
    elif fl and fl < 75_000_000: flags.append(f"Moderate float {fl/1e6:.1f}M shares")

    # ATR volatility flag
    if atr_pct >= 8:   flags.append(f"High volatility ATR {atr_pct:.1f}% of price — wider stops needed")
    elif atr_pct >= 4: flags.append(f"Moderate volatility ATR {atr_pct:.1f}% of price")

    hi, lo, px = data["52w_high"], data["52w_low"], data["price"]
    if hi and lo and hi != lo:
        rp = (px - lo) / (hi - lo) * 100
        if rp >= 90:   flags.append(f"Near 52-week high ({rp:.0f}% of range) — strong momentum")
        elif rp >= 70: flags.append(f"Upper range position ({rp:.0f}%)")
        elif rp <= 20: concerns.append(f"Near 52-week low ({rp:.0f}%) — downtrend risk")

    pead_signals = sum([gap >= 8, vm >= 2.0, si >= SHORT_INT_MIN])
    return {
        "flags":        flags,
        "concerns":     concerns,
        "pead_ready":   pead_signals >= 2,
        "squeeze_ready": si >= SHORT_INT_MIN and gap > 0,
        "signal_count": pead_signals,
    }


def filter_candidates(tickers):
    candidates = []
    for t in tickers:
        data = fetch_stock_data(t)
        if not data:
            continue
        quant = quant_score(data)
        if quant["signal_count"] >= 1 or quant["squeeze_ready"]:
            candidates.append({**data, "quant": quant})
    return candidates[:MAX_CANDIDATES]


# ══════════════════════════════════════════════════════════════════════════════
# AI BRIEFING — now includes ATR levels
# ══════════════════════════════════════════════════════════════════════════════

def build_briefing(stock: dict) -> str:
    q = stock["quant"]
    flags_text    = "\n".join(f"  ✓ {f}" for f in q["flags"])    or "  None"
    concerns_text = "\n".join(f"  ⚠ {c}" for c in q["concerns"]) or "  None"
    atr_pct = stock.get("atr_pct", 0)

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
  VWAP (today):    ${stock['vwap']}
  ATR-14:          ${stock['atr']} ({atr_pct:.1f}% of price)
  Today Volume:    {stock['volume']:,}
  20d Avg Volume:  {stock['avg_volume']:,}
  Volume Multiple: {stock['volume_mult']:.1f}x

Short Interest & Float:
  Short Interest:  {stock['short_pct']:.1f}% of float
  Float:           {f"{stock['float_shares']/1e6:.1f}M" if stock['float_shares'] else 'Unknown'} shares

Positioning:
  52-Week High:    ${stock['52w_high']}
  52-Week Low:     ${stock['52w_low']}

Quantitative Signals:
{flags_text}

Quantitative Concerns:
{concerns_text}

PEAD Candidate:    {'YES' if q['pead_ready'] else 'NO'}
Squeeze Candidate: {'YES' if q['squeeze_ready'] else 'NO'}

━━━ YOUR ANALYSIS TASK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are a seasoned professional trader. The ATR is ${stock['atr']} 
({atr_pct:.1f}% of price) — this means normal daily noise is 
approximately this wide. Factor this into your entry and stop thinking.

Assess this stock and respond ONLY in this exact JSON format:
{{
  "direction": "LONG" | "SHORT" | "NO TRADE",
  "thesis": "2-3 sentence explanation",
  "primary_driver": "PEAD" | "SQUEEZE" | "MOMENTUM" | "COMBINED" | "NONE",
  "timeframe": "e.g. 3-5 days",
  "key_risk": "the one thing that kills this trade",
  "conviction": "HIGH" | "MODERATE" | "LOW",
  "entry_confirmation": "specific price action or condition to wait for before entering — be precise",
  "entry_timing": "e.g. first 30 min, after 10:30am ET, on first pullback",
  "stop_reasoning": "why the ATR-based stop makes sense for this specific setup",
  "exit_rule": "what price action or condition signals the thesis has failed — beyond the stop"
}}
"""


def call_claude(briefing: str) -> dict | None:
    if not ANTHROPIC_API_KEY:
        log.warning("No ANTHROPIC_API_KEY — skipping AI call")
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
                "max_tokens": 800,
                "messages":   [{"role": "user", "content": briefing}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        text = text.replace("```json","").replace("```","").strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"AI JSON parse error: {e}")
        return None
    except Exception as e:
        log.error(f"Claude API error: {e}")
        return None


def analyse_candidates(candidates: list) -> list:
    signals = []
    for stock in candidates:
        log.info(f"  → AI analysing {stock['ticker']}...")
        briefing = build_briefing(stock)
        ai = call_claude(briefing)
        if not ai or ai.get("direction") == "NO TRADE":
            log.info(f"    {stock['ticker']}: NO TRADE")
            continue

        # Calculate all trade levels using ATR
        direction = ai["direction"]
        driver    = ai.get("primary_driver", "NONE")
        levels    = calc_trade_levels(stock, pd.DataFrame(), direction, driver)

        sig = {
            # Core data
            "ticker":         stock["ticker"],
            "name":           stock["name"],
            "sector":         stock["sector"],
            "price":          stock["price"],
            "prev_close":     stock["prev_close"],
            "gap_pct":        stock["gap_pct"],
            "volume_mult":    stock["volume_mult"],
            "short_pct":      stock["short_pct"],
            "float_shares":   stock["float_shares"],
            "52w_high":       stock["52w_high"],
            "52w_low":        stock["52w_low"],
            # ATR + levels
            "atr":            stock["atr"],
            "atr_pct":        stock["atr_pct"],
            "vwap":           stock["vwap"],
            "gap_open":       levels["gap_open"],
            # Trade management
            "stop_loss":      levels["stop_loss"],
            "stop_type":      levels["stop_type"],
            "risk_per_share": levels["risk_per_share"],
            "tp1":            levels["tp1"],
            "tp1_rr":         levels["tp1_rr"],
            "tp1_action":     levels["tp1_action"],
            "tp2":            levels["tp2"],
            "tp2_rr":         levels["tp2_rr"],
            "tp2_action":     levels["tp2_action"],
            "trailing_stop":  levels["trailing_stop_distance"],
            "entry_ideal":    levels["entry_ideal"],
            "entry_max":      levels["entry_max"],
            "entry_note_computed": levels["entry_note_computed"],
            # AI fields
            "direction":          direction,
            "thesis":             ai.get("thesis",""),
            "primary_driver":     driver,
            "timeframe":          ai.get("timeframe",""),
            "key_risk":           ai.get("key_risk",""),
            "conviction":         ai.get("conviction",""),
            "entry_confirmation": ai.get("entry_confirmation",""),
            "entry_timing":       ai.get("entry_timing",""),
            "stop_reasoning":     ai.get("stop_reasoning",""),
            "exit_rule":          ai.get("exit_rule",""),
            "quant_flags":        stock["quant"]["flags"],
            "timestamp":          datetime.now().isoformat(),
        }
        signals.append(sig)
        log.info(f"    {stock['ticker']}: {direction} | {ai.get('conviction')} | SL:${levels['stop_loss']} TP1:${levels['tp1']} TP2:${levels['tp2']}")
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# SCAN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def fetch_premarket_gappers():
    gappers = []
    for t in BASE_WATCHLIST:
        try:
            hist = yf.Ticker(t).history(period="2d", interval="1d")
            if len(hist) < 2: continue
            gap = ((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]) * 100
            if abs(gap) >= GAP_THRESHOLD_PCT:
                gappers.append(t)
        except Exception:
            pass
    return gappers


def run_scan():
    log.info("━━━ SCAN CYCLE STARTING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    universe = set(BASE_WATCHLIST)
    gappers  = fetch_premarket_gappers()
    universe.update(gappers)
    log.info(f"Universe: {len(universe)} | Gappers: {gappers}")

    candidates = filter_candidates(list(universe))
    log.info(f"Quant filter: {len(candidates)} candidates")

    signals = analyse_candidates(candidates)
    log.info(f"Signals issued: {len(signals)}")

    with STATE_LOCK:
        STATE["signals"]      = signals
        STATE["candidates"]   = [
            {"ticker": c["ticker"], "name": c["name"],
             "gap_pct": c["gap_pct"], "volume_mult": c["volume_mult"],
             "short_pct": c["short_pct"], "atr_pct": c.get("atr_pct",0),
             "quant_flags": c["quant"]["flags"]}
            for c in candidates
        ]
        STATE["watchlist"]    = list(universe)
        STATE["last_updated"] = datetime.now().isoformat()
        STATE["scan_count"]  += 1

    log.info("━━━ SCAN COMPLETE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def scan_loop():
    while True:
        try:
            run_scan()
        except Exception as e:
            log.error(f"Scan error: {e}")
            with STATE_LOCK:
                STATE["errors"].append({"time": datetime.now().isoformat(), "error": str(e)})
        time.sleep(SCAN_INTERVAL_SEC)


# ══════════════════════════════════════════════════════════════════════════════
# FLASK API
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

@app.route("/api/state")
def api_state():
    with STATE_LOCK:
        return jsonify(STATE)

@app.route("/api/signals")
def api_signals():
    with STATE_LOCK:
        return jsonify(STATE["signals"])

@app.route("/api/health")
def api_health():
    with STATE_LOCK:
        return jsonify({"status": "online", "scan_count": STATE["scan_count"], "last_updated": STATE["last_updated"]})

@app.route("/")
def index():
    return "<h2>MIS Engine v2 — Online ✓</h2><p>ATR stops + targets active</p>"


if __name__ == "__main__":
    log.info("▶ MIS Engine v2 starting...")
    log.info(f"  Port: {PORT} | AI key: {'SET ✓' if ANTHROPIC_API_KEY else 'NOT SET ✗'}")
    threading.Thread(target=scan_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
