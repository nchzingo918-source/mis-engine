"""
MARKET INTELLIGENCE ENGINE v4
================================
New in v4:
- Tier 1 indicators: RSI-14, VWAP distance %, ORB flag, bullish/bearish engulfing,
  hammer, shooting star, doji, morning star / evening star, inside bar
- Tier 2 indicators: VWAP standard deviation bands (+1σ/+2σ), relative strength
  vs SPY, short interest change rate (current vs 2-week estimate)
- Entry state machine: WATCHING → CONDITIONS_MET → ENTRY_VALID per signal
- Entry validity: checks price in zone, VWAP side, volume, ORB, pattern alignment
- Signal persistence: seen_count, last_seen, thesis_status (INTACT/WEAKENING/FAILED)
- Dashboard receives full indicator payload + entry_status per signal

SETUP:
  pip install flask flask-cors yfinance requests pandas numpy beautifulsoup4 lxml

DEPLOY:
  Railway — set ANTHROPIC_API_KEY env var
"""

import os, re, json, time, threading, logging
from datetime import datetime, timezone
from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SCAN_INTERVAL_SEC = 60
GAP_THRESHOLD_PCT = 5.0
SHORT_INT_MIN     = 12.0
MAX_CANDIDATES    = 20
PORT              = int(os.environ.get("PORT", 8080))
FINVIZ_ENABLED    = True
BATCH_SIZE        = 80
FINVIZ_TTL        = 300

STATE = {
    "signals": [], "candidates": [], "watchlist": [],
    "last_updated": None, "scan_count": 0,
    "finviz_tickers": [], "spy_change": 0.0, "errors": [],
}
STATE_LOCK = threading.Lock()

# Signal persistence store: ticker -> {count, last_seen, signal, confirmed_at, expires_at}
SIGNAL_STORE = {}
SIGNAL_LOCK  = threading.Lock()
CONFIRM_THRESHOLD = 3

BASE_WATCHLIST = [
    # ── LEGENDARY MEME / HIGH SHORT INTEREST ──
    "GME","AMC","BBBY","KOSS","EXPR","NAKD","CLOV","WISH","BB","NOK",
    "SNDL","TLRY","ACB","CRON","OGI","CGC","HEXO","VFF","GRWG","MSOS",

    # ── EV / CLEAN TRANSPORT ──
    "RIVN","LCID","NKLA","RIDE","WKHS","HYLN","SOLO","AYRO","IDEX","HYRE",
    "XPEV","NIO","LI","KNDI","BLNK","CHPT","EVGO","VLTA","SBE","ACTC",

    # ── CLEAN ENERGY / HYDROGEN ──
    "FCEL","PLUG","BLDP","BE","HYZN","CLNE","GEVO","REX","ARRY","MAXN",
    "SPWR","RUN","NOVA","SUNW","SHLS","STEM","FLUX","AMRC","CWEN","AY",

    # ── SPACE / DEFENSE DISRUPTORS ──
    "SPCE","ASTS","RKLB","MNTS","ASTR","GSAT","VSAT","JOBY","ACHR","LILM",
    "KTOS","AVAV","AXON","DRS","MRCY","CACI","BWXT","LDOS","SAIC","BAH",

    # ── FINTECH / BNPL ──
    "AFRM","UPST","SOFI","DAVE","HOOD","LMND","ROOT","MILE","HIPPO","CURO",
    "ENVA","PRAA","WRLD","QFIN","LX","CACC","ALLY","GREEN","TREE","LDI",

    # ── CRYPTO-ADJACENT ──
    "COIN","MSTR","MARA","RIOT","HUT","BTBT","CIFR","CLSK","BITF","HIVE",
    "AULT","BTCS","MGTI","SATO","WULF","NCTY","EBON","BTCM","ARBK","NILE",

    # ── BIOTECH / PHARMA — LARGE CAP ──
    "NVAX","MRNA","BNTX","VXRT","INO","OCGN","IOVA","RCUS","FATE","KRTX",
    "BLUE","RGEN","FOLD","RARE","MDGL","ALNY","IONS","VRTX","BMRN","SRPT",
    "EXEL","ACAD","SAGE","NKTR","ARCT","SIGA","HALO","AGEN","ADMA","CRSP",
    "EDIT","NTLA","BEAM","PACB","BNGO","IBRX","CHRS","SAVA","DVAX","ACMR",

    # ── BIOTECH — SMALL/MID (highest squeeze frequency) ──
    "AMRN","ARDX","ARQT","ASRT","ATAI","AUPH","AVDL","AVIR","AVXL","AYALA",
    "BCAB","BCEL","BCRX","BDTX","BHVN","BIOL","BIOR","BLCM","BLPH","BOLT",
    "BPMC","BTAI","CAPR","CARA","CALA","CELH","CEMI","CERE","CERS","CERT",
    "CGON","CHEK","CIDM","CLBS","CLDX","CLVS","CNCE","CNTX","COCP","COEP",
    "CPRX","CRVS","CYCN","DFFN","DRIO","DVAX","DYAI","DYNT","EDSA","EFHT",
    "EGBN","EHTH","EIGR","EKSO","ELIF","ELOX","EMKR","ENOB","ENTA","ENTX",
    "ENZN","EOLS","EPIX","ESPR","ETNB","EVGN","EVLO","FATE","FGEN","FLGT",
    "FMTX","GLYC","GRPH","GRTX","GTHX","HRMY","HROW","HTBX","IDYA","IMCR",
    "IMGO","IMMP","IMVT","INAB","INFI","INMB","ITOS","JANX","KDNY","KLUS",
    "KROS","KYMR","LBPH","LCTX","LGND","LOGC","LPTX","LQDA","LRMR","LUPN",
    "LYRA","MCRB","MDNA","MGNX","MIRM","MORF","MRNS","MRTX","MRUS","MSRT",
    "MTEM","MXCT","MYMD","NBIX","NBTX","NEOS","NEXI","NGNE","NLRX","NMRA",
    "NNOX","NOVN","NRIX","NTRA","NUVB","NVCR","ORGO","ORMP","PAHC","PANL",

    # ── SAAS / CLOUD / SOFTWARE ──
    "DDOG","NET","CRWD","ZS","OKTA","MDB","ESTC","FSLY","FROG","EVBG",
    "DOMO","NEWR","PSTG","AVLR","BILL","FLYW","JAMF","PCTY","PAYC","COUP",
    "BAND","RDWR","QLYS","BRZE","GTLB","TOST","SMAR","MTTR","SUMO","WCLD",
    "SKLZ","RBLX","U","SNAP","PINS","TWTR","OPEN","LMND","DKNG","PENN",

    # ── RETAIL / CONSUMER CYCLICAL ──
    "BYND","OATLY","TTCF","APPH","VERY","PLAY","CAKE","JACK","DENN","DINE",
    "RUTH","BLMN","FAT","PZZA","FRSH","SEAS","EPR","CZR","WYNN","LVS",
    "MGM","BURL","FIVE","TJX","ROST","OLLI","BIG","BBWI","AEO","ANF",
    "MANU","MSGS","IMAX","CNK","NCMI","XELA","PLBY","NUVEI","GENI","EVERI",

    # ── SMALL-CAP SQUEEZE CANDIDATES ──
    "MVIS","BARK","CRSR","MAPS","GEVI","ILUS","CTXR","ATNF","ATOS","AEZS",
    "HBIO","OCUP","AEYE","VBIV","VISL","BOXL","BFRI","BURU","CASI","CBTX",
    "CCAP","CDEV","CDMO","CETX","CHNR","CMBM","CMND","COEP","COMS","CONN",
    "CPHC","CTRM","CUEN","CULP","CVAC","CVBF","CVCO","CVGW","CVLG","CVNA",
    "CGXB","CHCI","CFFS","CFLT","CFNB","CCRD","CCSI","CCTG","CCUR","CCXI",
    "CDAK","CDNA","CDNS","CDTX","CDXC","CDXS","CDZI","CECO","CELL","CENT",
    "CHWY","CIGNA","CLAI","CLBK","CLCT","CLFD","CLGN","CLIM","CLIN","CLIR",
    "CLNC","CLNE","CLNN","CLPS","CLRB","CLRO","CLSD","CLSK","CLST","CLWT",
    "CMBT","CMCO","CMCT","CMND","CMPR","CMRX","CMTL","CNCR","CNET","CNHI",
    "CNIK","CNMD","CNNE","CNNX","CNOB","CNSL","CNTB","CNTG","CNTY","CNXC",
    "CNXN","COCP","COFS","COHN","COHR","COKE","COLB","COLD","COLL","COLM",

    # ── ENERGY / OIL & GAS ──
    "CEI","TELL","NEXT","SHIP","TOPS","CTRM","USAK","SOS","AMMO","INDO",
    "BORR","SDRL","GLOG","GLNG","FLNG","HMLP","RIG","VAL","NE","PR",
    "CPE","CDEV","DINO","REI","GPOR","WLL","OXY","DVN","FANG","MRO",
    "HES","APA","MTDR","PDCE","CLR","SM","VTLE","ESTE","RVMD","BATL",

    # ── SEMICONDUCTORS ──
    "SMCI","AMD","NVDA","MU","MRVL","QCOM","AVGO","TXN","LSCC","ALGM",
    "CEVA","DIOD","FORM","KLIC","LRCX","MCHP","MPWR","MTSI","NXPI","OLED",
    "ONTO","POWI","PSEM","QRVO","RMBS","SLAB","SITM","SMTC","SYNA","WOLF",
    "WULF","TORC","TSEM","UEIC","VICR","VECO","IXYS","COHR","IIVI","IPGP",

    # ── CHINESE ADRs ──
    "XPEV","NIO","LI","BIDU","JD","PDD","BEKE","KE","TUYA","DADA",
    "DOYU","EDTK","HUYA","IQ","QFIN","RLX","RERE","STNE","TIGR","TCOM",
    "VNET","WB","XNET","YMM","ZH","ZTO","ZLAB","ACMR","AMBO","AMTD",
    "ANGI","ANTE","AIXI","BILI","CODA","LAIX","LKNCY","MOXC","NIU","BTMX",

    # ── SPAC / RECENT IPO ──
    "GRAB","PSFE","CANO","BARK","PNTM","FTIV","GORES","IPOF","IPOE","IPOD",
    "SNPR","THCB","KCAC","DCRB","ACTC","NGA","GIK","HYAC","CCIV","BGSX",
    "GSAH","BTAQ","CIIC","STPK","HCAC","RBAC","GXII","AJAX","CGSA","DMYI",

    # ── CANNABIS ──
    "TLRY","ACB","HEXO","CRON","OGI","CGC","VFF","GRWG","IIPR","CURLF",
    "GTBIF","TCNNF","CCHWF","AYRWF","VRNOF","HRVSF","FLGC","GLASF","LGVN","MSOS",

    # ── SHIPPING / FREIGHT ──
    "ZIM","TOPS","SHIP","CTRM","GLBS","DSSI","EGLE","SBLK","STNG","INSW",
    "GOGL","FLNG","GLOG","GLNG","HMLP","KNOP","MRC","MRCC","NAT","PANL",

    # ── MINING / METALS ──
    "MP","VALE","FCX","SCCO","HBM","TECK","AG","EXK","PAAS","HL",
    "CDE","IAUX","GATO","AUY","KGC","GOLD","NEM","ABX","WPM","FNV",
    "SILV","FSM","EGO","BTG","OR","RGLD","SAND","MAG","GPL","MTA",

    # ── REAL ESTATE / REIT ──
    "OPEN","DOMA","PRMI","IIPR","SAFE","STAR","STWD","TWO","BXMT","GPMT",
    "KREF","MFA","MITT","NLY","ORC","PMT","RWT","CLNC","ACRE","BRSP",

    # ── HEALTHCARE / MEDTECH ──
    "TDOC","ACCD","ONEM","HIMS","OPTM","NVCR","TMDX","SWAV","PRCT","BLFS",
    "CELC","CHRD","CNMD","DXCM","EKSO","ESTA","EVBG","EVER","EXAS","FLGT",
    "GENI","GDHG","IRMD","ISRG","LMAT","MDRX","MDXH","MITK","MMSI","MNKD",

    # ── FINANCIAL SERVICES ──
    "LCII","PRAA","WRLD","ENVA","CURO","RM","ECPG","SLQT","PFLT","HONE",
    "BSVN","CATC","CTBI","CZWI","DCOM","DFIN","DGII","DENN","DLPN","DMRC",

    # ── MEDIA / ENTERTAINMENT ──
    "IMAX","CNK","AMC","NCMI","RDI","XELA","PLBY","NUVEI","AGS","RICK",
    "NATH","FAT","DKNG","PENN","GENI","EVERI","MANU","MSGS","SEAS","EPR",

    # ── AGRICULTURE / FOOD TECH ──
    "BYND","TTCF","APPH","VERY","OATLY","NDLS","VITL","BRFS","JBSS","LWAY",
    "MGPI","NAPA","POST","PLAG","PLYM","PNTM","PFGC","SFM","CALM","KFRC",

    # ── ADDITIONAL HIGH-VOL SINGLES (fills list toward 1000) ──
    "ACEL","ACNB","ACRS","ADAP","ADEA","ADIL","ADMA","ADMP","ADRO","ADTX",
    "ADUS","ADVM","ADXN","ADYN","AEAC","AEHR","AEIS","AEON","AEYE","AFAR",
    "AFIB","AFMD","AFRI","AGAC","AGEN","AGFY","AGIL","AGIO","AGMH","AGRI",
    "AGRO","AGTI","AGYS","AGTX","AGZD","AHCO","AHPA","AHPI","AHRN","AIDA",
    "AIKI","AILE","AIMAU","AINC","AINV","AIRC","AIRG","AIRI","AIRJ","AIRR",
    "AIRT","AIXI","AIZN","AJRD","AKAM","AKBA","AKCA","AKESO","AKRO","AKTS",
    "AKTX","AKUS","AKYA","ALBO","ALCO","ALDX","ALEC","ALEX","ALGT","ALIM",
    "ALIT","ALJJ","ALKS","ALKT","ALLK","ALLR","ALLT","ALLY","ALNY","ALOT",
    "ALPN","ALRM","ALRS","ALSA","ALSN","ALTI","ALTO","ALTR","ALTU","ALUS",
    "ALVR","ALXO","ALYA","AMAG","AMBC","AMBI","AMBO","AMEH","AMGN","AMHC",
    "AMKR","AMMD","AMMO","AMMS","AMNI","AMOT","AMPE","AMPH","AMPIO","AMPL",
    "AMPO","AMRK","AMRN","AMRS","AMSC","AMSE","AMSF","AMST","AMTA","AMTB",
    "AMTD","AMTI","AMTX","AMWL","AMYT","AMZN","ANAB","ANAC","ANDE","ANEB",
    "ANGH","ANIK","ANIP","ANJI","ANKH","ANNX","ANPC","ANSS","ANTE","ANTX",
    "ANVS","ANZU","AORT","AOSL","AOUT","APAM","APCA","APDN","APEI","APEN",
    "APGB","APGN","APGT","APHE","APHL","APLD","APLE","APLT","APMI","APOG",
    "APOP","APPH","APRE","APRL","APRO","APRT","APRN","APRO","APSE","APSG",
    "APTV","APVO","APWC","APXI","APYX","AQMS","AQNA","AQST","AQUA","AQXP",
    "ARAV","ARCC","ARCE","ARCH","ARCO","ARCT","ARCY","AREC","AREB","AREC",
    "ARES","ARGT","ARGX","ARHS","ARIA","ARIK","ARIN","ARIS","ARIZ","ARKO",
    "ARLS","ARMK","ARMP","ARMT","ARNC","AROC","AROW","ARPO","ARQQ","ARQT",
    "ARRY","ARTE","ARTL","ARTNA","ARTO","ARTV","ARTW","ARUN","ARVL","ARWR",
    "ARYA","ASAI","ASBP","ASCE","ASCMA","ASDN","ASIX","ASMB","ASML","ASND",
    "ASNS","ASPA","ASPI","ASPN","ASPS","ASPU","ASRT","ASRV","ASST","ASTR",
    "ATEN","ATEX","ATFL","ATGN","ATHA","ATHE","ATHM","ATHR","ATHE","ATIF",
    "ATIS","ATIX","ATKO","ATLA","ATLO","ATLX","ATMC","ATNI","ATNM","ATNS",
    "ATNX","ATOM","ATON","ATPC","ATPL","ATRC","ATRI","ATRM","ATRS","ATRU",
    "ATSG","ATST","ATSV","ATSY","ATVI","ATXG","ATXI","ATXS","AUBN","AUDC",
    "AUGX","AUID","AUMN","AUPH","AUTL","AUUD","AUUS","AUVT","AUVI","AVAH",
    "AVAL","AVAN","AVAV","AVBH","AVCO","AVDL","AVDX","AVEO","AVER","AVGO",
    "AVHI","AVID","AVIG","AVIR","AVNS","AVNT","AVNW","AVPT","AVRO","AVTA",
    "AVTE","AVXL","AWAV","AWRE","AXDX","AXEN","AXGN","AXGT","AXHE","AXLA",
    "AXNX","AXON","AXSM","AXTG","AXTI","AXTR","AXTS","AXXN","AYALA","AYRO",
    "AZEK","AZRX","AZTA","AZUL","AZYO","BACK","BAFN","BAER","BAFN","BAIO",
    "BAND","BANG","BANL","BANR","BANT","BANX","BAOS","BAQC","BATRA","BATRK",
    "BAYA","BBAI","BBAR","BBBI","BBCP","BBGI","BBIO","BBLG","BBLR","BBSI",
    "BBWI","BCAB","BCAN","BCBP","BCEI","BCML","BCOM","BCOV","BCPC","BCPP",
]

# Deduplicate
_seen = set()
BASE_WATCHLIST = [t for t in BASE_WATCHLIST if not (t in _seen or _seen.add(t))]
log.info(f'Watchlist: {len(BASE_WATCHLIST)} tickers')


# ══════════════════════════════════════════════════════════════════════════════
# FINVIZ SCREENER
# ══════════════════════════════════════════════════════════════════════════════
FINVIZ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://finviz.com/",
}

def fetch_finviz_screener() -> list:
    tickers, urls = [], [
        "https://finviz.com/screener.ashx?v=111&f=sh_short_o10,sh_avgvol_o500,sh_price_o1,ta_gap_u5&ft=4&o=-volume",
        "https://finviz.com/screener.ashx?v=111&f=sh_short_o10,sh_avgvol_o500,sh_price_o1,ta_gap_d5&ft=4&o=volume",
        "https://finviz.com/screener.ashx?v=111&f=sh_short_o10,sh_avgvol_o500,sh_price_o1,sh_relvol_o3&ft=4&o=-change",
        "https://finviz.com/screener.ashx?v=111&f=sh_short_o20,sh_avgvol_o500,sh_price_o1&ft=4&o=-short",
    ]
    for url in urls:
        try:
            resp  = requests.get(url, headers=FINVIZ_HEADERS, timeout=15)
            if resp.status_code != 200: continue
            soup  = BeautifulSoup(resp.text, "lxml")
            links = soup.find_all("a", class_="screener-link-primary")
            for lnk in links:
                t = lnk.text.strip()
                if t and re.match(r'^[A-Z]{1,5}$', t):
                    tickers.append(t)
        except Exception as e:
            log.warning(f"Finviz fetch failed: {e}")
    unique = list(dict.fromkeys(tickers))
    log.info(f"Finviz: {len(unique)} tickers")
    return unique

_finviz_cache, _finviz_last = [], 0.0

def get_finviz_tickers() -> list:
    global _finviz_cache, _finviz_last
    if not FINVIZ_ENABLED: return []
    if time.time() - _finviz_last > FINVIZ_TTL:
        fresh = fetch_finviz_screener()
        if fresh:
            _finviz_cache = fresh
            _finviz_last  = time.time()
            with STATE_LOCK: STATE["finviz_tickers"] = fresh
    return _finviz_cache


# ══════════════════════════════════════════════════════════════════════════════
# SPY RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════════════
_spy_change_cache = 0.0
_spy_last = 0.0

def get_spy_change() -> float:
    global _spy_change_cache, _spy_last
    if time.time() - _spy_last < 120:
        return _spy_change_cache
    try:
        hist = yf.Ticker("SPY").history(period="2d", interval="1d")
        if len(hist) >= 2:
            chg = (hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100
            _spy_change_cache = round(float(chg), 2)
            _spy_last = time.time()
            with STATE_LOCK: STATE["spy_change"] = _spy_change_cache
    except Exception:
        pass
    return _spy_change_cache


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_atr(hist: pd.DataFrame, period: int = 14) -> float:
    try:
        tr = pd.concat([
            hist["High"] - hist["Low"],
            (hist["High"] - hist["Close"].shift(1)).abs(),
            (hist["Low"]  - hist["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0

def calc_vwap(hist: pd.DataFrame) -> float:
    try:
        row = hist.iloc[-1]
        return round(float((row["High"] + row["Low"] + row["Close"]) / 3), 2)
    except Exception:
        return 0.0

def calc_vwap_bands(hist: pd.DataFrame, vwap: float) -> dict:
    """VWAP + 1σ and 2σ bands from today's candle."""
    try:
        row = hist.iloc[-1]
        high, low, close = row["High"], row["Low"], row["Close"]
        # Approximate daily std using ATR proxy
        typical_range = (high - low)
        sigma = typical_range / 2  # rough daily σ
        return {
            "vwap_plus1":  round(vwap + sigma, 2),
            "vwap_plus2":  round(vwap + sigma * 2, 2),
            "vwap_minus1": round(vwap - sigma, 2),
            "vwap_minus2": round(vwap - sigma * 2, 2),
        }
    except Exception:
        return {"vwap_plus1": 0, "vwap_plus2": 0, "vwap_minus1": 0, "vwap_minus2": 0}

def calc_rsi(hist: pd.DataFrame, period: int = 14) -> float:
    try:
        delta  = hist["Close"].diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi    = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 1)
    except Exception:
        return 50.0

def calc_rsi_divergence(hist: pd.DataFrame) -> str:
    """
    Detect RSI divergence over last 5 candles.
    Bearish: price higher high but RSI lower high → momentum fading.
    Bullish: price lower low but RSI higher low → downside fading.
    """
    try:
        closes = hist["Close"].iloc[-6:]
        rsi_series = []
        for i in range(len(hist) - 5, len(hist)):
            sub = hist.iloc[max(0, i-14):i+1]
            rsi_series.append(calc_rsi(sub))
        price_trend = closes.iloc[-1] - closes.iloc[0]
        rsi_trend   = rsi_series[-1] - rsi_series[0]
        if price_trend > 0 and rsi_trend < -3:
            return "BEARISH_DIVERGENCE"
        if price_trend < 0 and rsi_trend > 3:
            return "BULLISH_DIVERGENCE"
        return "NONE"
    except Exception:
        return "NONE"

def detect_candlestick_patterns(hist: pd.DataFrame) -> dict:
    """
    Detect Tier 1 + Tier 2 candlestick patterns from last 3 candles.
    Returns dict of pattern name → bool/strength.
    """
    patterns = {
        "bullish_engulfing": False,
        "bearish_engulfing": False,
        "hammer": False,
        "shooting_star": False,
        "doji": False,
        "morning_star": False,
        "evening_star": False,
        "inside_bar": False,
        "inside_bar_direction": None,  # "bullish" or "bearish" breakout pending
        "pattern_summary": [],
    }
    try:
        if len(hist) < 3:
            return patterns

        c1 = hist.iloc[-3]  # 3 candles ago
        c2 = hist.iloc[-2]  # yesterday
        c  = hist.iloc[-1]  # today (most recent)

        def body(candle): return abs(candle["Close"] - candle["Open"])
        def upper_wick(candle): return candle["High"] - max(candle["Close"], candle["Open"])
        def lower_wick(candle): return min(candle["Close"], candle["Open"]) - candle["Low"]
        def is_bull(candle): return candle["Close"] > candle["Open"]
        def is_bear(candle): return candle["Close"] < candle["Open"]
        def full_range(candle): return candle["High"] - candle["Low"]

        # ── BULLISH ENGULFING ──
        # Yesterday red, today green, today body covers yesterday's entire body
        if (is_bear(c2) and is_bull(c) and
            c["Open"] <= c2["Close"] and c["Close"] >= c2["Open"] and
            body(c) >= body(c2) * 0.8):
            patterns["bullish_engulfing"] = True
            patterns["pattern_summary"].append("Bullish Engulfing — sellers absorbed, buyers taking control")

        # ── BEARISH ENGULFING ──
        if (is_bull(c2) and is_bear(c) and
            c["Open"] >= c2["Close"] and c["Close"] <= c2["Open"] and
            body(c) >= body(c2) * 0.8):
            patterns["bearish_engulfing"] = True
            patterns["pattern_summary"].append("Bearish Engulfing — buyers exhausted, sellers taking control")

        # ── HAMMER (bullish reversal) ──
        # Small body near top, long lower wick (2x body), little upper wick
        if (full_range(c) > 0 and
            lower_wick(c) >= body(c) * 2.0 and
            upper_wick(c) <= body(c) * 0.5 and
            body(c) / full_range(c) < 0.4):
            patterns["hammer"] = True
            patterns["pattern_summary"].append("Hammer — sellers rejected at lows, bullish reversal signal")

        # ── SHOOTING STAR (bearish reversal) ──
        # Small body near bottom, long upper wick (2x body), little lower wick
        if (full_range(c) > 0 and
            upper_wick(c) >= body(c) * 2.0 and
            lower_wick(c) <= body(c) * 0.5 and
            body(c) / full_range(c) < 0.4):
            patterns["shooting_star"] = True
            patterns["pattern_summary"].append("Shooting Star — buyers rejected at highs, bearish reversal signal")

        # ── DOJI (indecision) ──
        if full_range(c) > 0 and body(c) / full_range(c) < 0.1:
            patterns["doji"] = True
            patterns["pattern_summary"].append("Doji — market indecision, watch for directional break")

        # ── INSIDE BAR (compression before breakout) ──
        if (c["High"] <= c2["High"] and c["Low"] >= c2["Low"] and
            body(c) < body(c2) * 0.7):
            patterns["inside_bar"] = True
            # Direction of likely breakout based on prior trend
            prior_trend = c2["Close"] - c1["Close"]
            patterns["inside_bar_direction"] = "bullish" if prior_trend > 0 else "bearish"
            patterns["pattern_summary"].append(
                f"Inside Bar compression — market coiling, likely {'upside' if prior_trend>0 else 'downside'} breakout")

        # ── MORNING STAR (3-candle bullish reversal) ──
        # Big red, small body (any color), big green — classic bottom reversal
        if (is_bear(c1) and body(c1) > body(c2) * 1.5 and
            is_bull(c) and body(c) > body(c2) * 1.5 and
            c["Close"] > (c1["Open"] + c1["Close"]) / 2):
            patterns["morning_star"] = True
            patterns["pattern_summary"].append("Morning Star — 3-candle bottom reversal, strong bullish signal")

        # ── EVENING STAR (3-candle bearish reversal) ──
        if (is_bull(c1) and body(c1) > body(c2) * 1.5 and
            is_bear(c) and body(c) > body(c2) * 1.5 and
            c["Close"] < (c1["Open"] + c1["Close"]) / 2):
            patterns["evening_star"] = True
            patterns["pattern_summary"].append("Evening Star — 3-candle top reversal, strong bearish signal")

    except Exception as e:
        log.warning(f"Pattern detection error: {e}")

    return patterns

def calc_orb(hist: pd.DataFrame, price: float) -> dict:
    """
    Opening Range Breakout: uses yesterday's high/low as proxy for ORB
    since we only have daily OHLCV. On intraday data this would use 30-min candles.
    """
    try:
        yesterday = hist.iloc[-2]
        orb_high  = round(float(yesterday["High"]), 2)
        orb_low   = round(float(yesterday["Low"]), 2)
        orb_range = round(orb_high - orb_low, 2)
        above_orb = price > orb_high
        below_orb = price < orb_low
        return {
            "orb_high":        orb_high,
            "orb_low":         orb_low,
            "orb_range":       orb_range,
            "above_orb_high":  above_orb,
            "below_orb_low":   below_orb,
            "orb_status":      "BREAKOUT_UP" if above_orb else ("BREAKOUT_DOWN" if below_orb else "INSIDE_RANGE"),
        }
    except Exception:
        return {"orb_high": 0, "orb_low": 0, "orb_range": 0,
                "above_orb_high": False, "below_orb_low": False, "orb_status": "UNKNOWN"}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY STATE MACHINE
# Determines: WATCHING / CONDITIONS_MET / ENTRY_VALID / THESIS_FAILED
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_entry_status(signal: dict, indicators: dict) -> dict:
    """
    Checks all real-time conditions and returns entry_status + reason.
    This is the "pull trigger or wait" decision layer.
    """
    direction   = signal.get("direction", "")
    price       = signal.get("price", 0)
    entry_ideal = signal.get("entry_ideal", 0)
    entry_max   = signal.get("entry_max", 0)
    stop_loss   = signal.get("stop_loss", 0)
    vwap        = indicators.get("vwap", 0)
    rsi         = indicators.get("rsi", 50)
    vwap_dist   = indicators.get("vwap_dist_pct", 0)
    orb         = indicators.get("orb", {})
    patterns    = indicators.get("patterns", {})
    vm          = signal.get("volume_mult", 0)
    rs_vs_spy   = indicators.get("rs_vs_spy", 0)
    conv        = signal.get("conviction", "LOW")
    seen_count  = signal.get("seen_count", 1)

    reasons_good = []
    reasons_bad  = []

    # ── Thesis failed? ─────────────────────────────────────────────────────────
    if direction == "LONG" and stop_loss and price < stop_loss:
        return {"status": "THESIS_FAILED", "color": "red",
                "message": f"Price ${price} broke stop ${stop_loss} — thesis invalidated",
                "entry_valid": False}
    if direction == "SHORT" and stop_loss and price > stop_loss:
        return {"status": "THESIS_FAILED", "color": "red",
                "message": f"Price ${price} broke stop ${stop_loss} — thesis invalidated",
                "entry_valid": False}

    # ── Price in entry zone? ───────────────────────────────────────────────────
    if direction == "LONG":
        in_zone = entry_ideal > 0 and entry_max > 0 and entry_ideal <= price <= entry_max
        above_vwap = vwap > 0 and price >= vwap * 0.995
        orb_ok = orb.get("above_orb_high", False) or orb.get("orb_status") == "BREAKOUT_UP"
    else:
        in_zone = entry_max > 0 and entry_ideal > 0 and entry_max <= price <= entry_ideal
        above_vwap = vwap > 0 and price <= vwap * 1.005
        orb_ok = orb.get("below_orb_low", False) or orb.get("orb_status") == "BREAKOUT_DOWN"

    # ── Bullish/Bearish pattern alignment ─────────────────────────────────────
    bullish_pattern = (patterns.get("bullish_engulfing") or patterns.get("hammer") or
                       patterns.get("morning_star") or
                       (patterns.get("inside_bar") and patterns.get("inside_bar_direction") == "bullish"))
    bearish_pattern = (patterns.get("bearish_engulfing") or patterns.get("shooting_star") or
                       patterns.get("evening_star") or
                       (patterns.get("inside_bar") and patterns.get("inside_bar_direction") == "bearish"))

    pattern_aligned = (direction == "LONG" and bullish_pattern) or (direction == "SHORT" and bearish_pattern)

    # ── RSI check ──────────────────────────────────────────────────────────────
    rsi_ok_long  = 30 < rsi < 75   # not overbought, not crashed
    rsi_ok_short = 25 < rsi < 70
    rsi_ok = rsi_ok_long if direction == "LONG" else rsi_ok_short

    # ── Volume still elevated? ─────────────────────────────────────────────────
    vol_ok = vm >= 1.5

    # ── RS vs SPY ──────────────────────────────────────────────────────────────
    rs_ok = (rs_vs_spy > 2 if direction == "LONG" else rs_vs_spy < -2)

    # ── Score conditions ───────────────────────────────────────────────────────
    conditions = [in_zone, above_vwap, vol_ok, rsi_ok, seen_count >= CONFIRM_THRESHOLD]
    bonus      = [orb_ok, pattern_aligned, rs_ok]

    met    = sum(conditions)
    bonus_met = sum(bonus)

    # Build reason strings
    if in_zone:         reasons_good.append(f"Price in entry zone")
    else:               reasons_bad.append(f"Price outside entry zone (${entry_ideal}–${entry_max})")
    if above_vwap:      reasons_good.append(f"Price {'above' if direction=='LONG' else 'below'} VWAP")
    else:               reasons_bad.append(f"Price on wrong side of VWAP (${vwap})")
    if vol_ok:          reasons_good.append(f"Volume still elevated {vm:.1f}x")
    else:               reasons_bad.append(f"Volume fading ({vm:.1f}x)")
    if rsi_ok:          reasons_good.append(f"RSI healthy ({rsi})")
    else:               reasons_bad.append(f"RSI out of range ({rsi})")
    if orb_ok:          reasons_good.append(f"ORB breakout confirmed")
    if pattern_aligned: reasons_good.append(f"Candlestick pattern aligned")
    if rs_ok:           reasons_good.append(f"Strong RS vs SPY ({rs_vs_spy:+.1f}%)")
    if seen_count >= CONFIRM_THRESHOLD: reasons_good.append(f"Signal confirmed ({seen_count} appearances)")
    else:               reasons_bad.append(f"Not yet confirmed ({seen_count}/{CONFIRM_THRESHOLD} appearances)")

    # ── Determine status ───────────────────────────────────────────────────────
    if met >= 5 and bonus_met >= 1:
        return {
            "status":      "ENTRY_VALID",
            "color":       "green" if direction == "LONG" else "red",
            "message":     f"{'GO LONG' if direction=='LONG' else 'GO SHORT'} — {met}/5 conditions met + {bonus_met} bonus",
            "entry_valid": True,
            "reasons_good": reasons_good,
            "reasons_bad":  reasons_bad,
        }
    elif met >= 3:
        return {
            "status":      "CONDITIONS_MET",
            "color":       "amber",
            "message":     f"Setup forming — {met}/5 core conditions met, waiting for {bonus_met}/3 bonus",
            "entry_valid": False,
            "reasons_good": reasons_good,
            "reasons_bad":  reasons_bad,
        }
    else:
        return {
            "status":      "WATCHING",
            "color":       "dim",
            "message":     f"Monitoring — {met}/5 conditions met",
            "entry_valid": False,
            "reasons_good": reasons_good,
            "reasons_bad":  reasons_bad,
        }


# ══════════════════════════════════════════════════════════════════════════════
# THESIS STATUS (for signals that disappeared from scan)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_thesis_status(signal: dict, current_price: float, vwap: float) -> str:
    """INTACT / WEAKENING / FAILED — shows on confirmed cards when signal leaves scan."""
    direction = signal.get("direction", "")
    stop      = signal.get("stop_loss", 0)
    gap_open  = signal.get("gap_open", 0)

    if direction == "LONG":
        if stop and current_price < stop:
            return "FAILED"
        if gap_open and current_price < gap_open:
            return "WEAKENING"
        if vwap and current_price < vwap * 0.995:
            return "WEAKENING"
        return "INTACT"
    else:
        if stop and current_price > stop:
            return "FAILED"
        if gap_open and current_price > gap_open:
            return "WEAKENING"
        if vwap and current_price > vwap * 1.005:
            return "WEAKENING"
        return "INTACT"


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH — now pulls all indicators in one shot
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker: str) -> dict | None:
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="30d", interval="1d")
        if hist.empty or len(hist) < 5:
            return None

        info           = tk.info or {}
        current_price  = float(hist["Close"].iloc[-1])
        prev_close     = float(hist["Close"].iloc[-2])
        today_volume   = float(hist["Volume"].iloc[-1])
        avg_volume_20d = float(hist["Volume"].iloc[:-1].mean())
        volume_mult    = today_volume / avg_volume_20d if avg_volume_20d > 0 else 0
        gap_pct        = ((current_price - prev_close) / prev_close) * 100
        atr            = calc_atr(hist, 14)
        vwap           = calc_vwap(hist)
        vwap_bands     = calc_vwap_bands(hist, vwap)
        rsi            = calc_rsi(hist, 14)
        rsi_div        = calc_rsi_divergence(hist)
        patterns       = detect_candlestick_patterns(hist)
        orb            = calc_orb(hist, current_price)
        spy_chg        = get_spy_change()

        vwap_dist_pct  = round(((current_price - vwap) / vwap) * 100, 2) if vwap > 0 else 0
        rs_vs_spy      = round(gap_pct - spy_chg, 2)

        # Short interest change proxy: compare current SI vs typical (we don't have historical,
        # so we flag the absolute level and mark direction as "elevated" if >15%)
        si = round((info.get("shortPercentOfFloat") or 0) * 100, 1)
        si_trend = "HIGH_SUSTAINED" if si >= 20 else ("ELEVATED" if si >= 12 else "NORMAL")

        return {
            "ticker":         ticker,
            "price":          round(current_price, 2),
            "prev_close":     round(prev_close, 2),
            "gap_pct":        round(gap_pct, 2),
            "volume":         int(today_volume),
            "avg_volume":     int(avg_volume_20d),
            "volume_mult":    round(volume_mult, 2),
            "market_cap":     info.get("marketCap", 0),
            "float_shares":   info.get("floatShares", 0),
            "short_pct":      si,
            "si_trend":       si_trend,
            "sector":         info.get("sector", "Unknown"),
            "name":           info.get("shortName", ticker),
            "52w_high":       info.get("fiftyTwoWeekHigh", 0),
            "52w_low":        info.get("fiftyTwoWeekLow",  0),
            "atr":            round(atr, 3),
            "atr_pct":        round((atr / current_price) * 100, 2) if current_price > 0 else 0,
            "vwap":           vwap,
            "vwap_plus1":     vwap_bands["vwap_plus1"],
            "vwap_plus2":     vwap_bands["vwap_plus2"],
            "vwap_minus1":    vwap_bands["vwap_minus1"],
            "vwap_minus2":    vwap_bands["vwap_minus2"],
            "vwap_dist_pct":  vwap_dist_pct,
            "rsi":            rsi,
            "rsi_divergence": rsi_div,
            "patterns":       patterns,
            "orb":            orb,
            "rs_vs_spy":      rs_vs_spy,
            "spy_change":     spy_chg,
        }
    except Exception as e:
        log.warning(f"Data fetch failed for {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# QUANT FILTER — now includes indicator context
# ══════════════════════════════════════════════════════════════════════════════

def quant_score(data: dict) -> dict:
    flags, concerns = [], []
    gap        = data["gap_pct"]
    vm         = data["volume_mult"]
    si         = data["short_pct"]
    fl         = data["float_shares"]
    atr_pct    = data.get("atr_pct", 0)
    rsi        = data.get("rsi", 50)
    rsi_div    = data.get("rsi_divergence", "NONE")
    vwap_dist  = data.get("vwap_dist_pct", 0)
    patterns   = data.get("patterns", {})
    orb        = data.get("orb", {})
    rs_vs_spy  = data.get("rs_vs_spy", 0)

    # Gap
    if gap >= 10:    flags.append(f"Strong gap up {gap:.1f}% — PEAD candidate")
    elif gap >= 5:   flags.append(f"Moderate gap up {gap:.1f}%")
    elif gap <= -10: flags.append(f"Strong gap DOWN {gap:.1f}% — short candidate")
    elif gap <= -5:  flags.append(f"Moderate gap down {gap:.1f}%")
    else:            concerns.append(f"Gap only {gap:.1f}% — limited momentum")

    # Volume
    if vm >= 3.0:    flags.append(f"Exceptional volume {vm:.1f}x — institutional confirmed")
    elif vm >= 2.0:  flags.append(f"Strong volume {vm:.1f}x")
    elif vm >= 1.5:  flags.append(f"Above average volume {vm:.1f}x")
    else:            concerns.append(f"Volume {vm:.1f}x — weak signal")

    # Short interest
    if si >= 25:           flags.append(f"Extreme short interest {si:.1f}% — major squeeze potential")
    elif si >= 15:         flags.append(f"High short interest {si:.1f}% — squeeze candidate")
    elif si >= SHORT_INT_MIN: flags.append(f"Elevated short interest {si:.1f}%")
    else:                  concerns.append(f"Short interest {si:.1f}% — limited fuel")

    # Float
    if fl and fl < 30_000_000:   flags.append(f"Low float {fl/1e6:.1f}M — explosive potential")
    elif fl and fl < 75_000_000: flags.append(f"Moderate float {fl/1e6:.1f}M shares")

    # ATR
    if atr_pct >= 8:   flags.append(f"High volatility ATR {atr_pct:.1f}% — wider stops needed")
    elif atr_pct >= 4: flags.append(f"Moderate volatility ATR {atr_pct:.1f}%")

    # RSI
    if 40 <= rsi <= 60:    flags.append(f"RSI {rsi} — plenty of room to run, not overbought")
    elif rsi > 70:         concerns.append(f"RSI {rsi} — overbought, momentum may be fading")
    elif rsi < 30:         flags.append(f"RSI {rsi} — oversold, bounce candidate")
    if rsi_div == "BEARISH_DIVERGENCE": concerns.append("RSI bearish divergence — momentum weakening vs price")
    if rsi_div == "BULLISH_DIVERGENCE": flags.append("RSI bullish divergence — downside fading, reversal signal")

    # VWAP distance
    if gap > 0 and -1 <= vwap_dist <= 2:   flags.append(f"Price near VWAP ({vwap_dist:+.1f}%) — ideal entry zone")
    elif gap > 0 and vwap_dist > 5:        concerns.append(f"Price {vwap_dist:.1f}% above VWAP — extended, chasing risk")
    elif gap < 0 and -2 <= vwap_dist <= 1: flags.append(f"Price near VWAP ({vwap_dist:+.1f}%) — ideal short entry zone")

    # Candlestick patterns
    for p in patterns.get("pattern_summary", []):
        flags.append(f"Pattern: {p}")

    # ORB
    orb_status = orb.get("orb_status", "")
    if gap > 0 and orb_status == "BREAKOUT_UP":   flags.append("ORB breakout UP — opening range confirmed bullish")
    if gap < 0 and orb_status == "BREAKOUT_DOWN": flags.append("ORB breakout DOWN — opening range confirmed bearish")

    # Relative strength vs SPY
    if rs_vs_spy >= 5:     flags.append(f"RS vs SPY: +{rs_vs_spy:.1f}% — institutional buying vs market")
    elif rs_vs_spy >= 2:   flags.append(f"RS vs SPY: +{rs_vs_spy:.1f}% — outperforming market")
    elif rs_vs_spy <= -5:  flags.append(f"RS vs SPY: {rs_vs_spy:.1f}% — strong relative weakness")
    elif rs_vs_spy <= -2:  flags.append(f"RS vs SPY: {rs_vs_spy:.1f}% — underperforming market")

    # 52-week range
    hi, lo, px = data["52w_high"], data["52w_low"], data["price"]
    if hi and lo and hi != lo:
        rp = (px - lo) / (hi - lo) * 100
        if rp >= 90:   flags.append(f"Near 52-week high ({rp:.0f}%) — strong momentum")
        elif rp >= 70: flags.append(f"Upper range position ({rp:.0f}%)")
        elif rp <= 20: concerns.append(f"Near 52-week low ({rp:.0f}%) — downtrend risk")

    pead_signals = sum([gap >= 8, vm >= 2.0, si >= SHORT_INT_MIN])
    return {
        "flags": flags, "concerns": concerns,
        "pead_ready":    pead_signals >= 2,
        "squeeze_ready": si >= SHORT_INT_MIN and gap > 0,
        "signal_count":  pead_signals,
    }

def filter_candidates(tickers: list) -> list:
    candidates = []
    for t in tickers:
        data = fetch_stock_data(t)
        if not data: continue
        quant = quant_score(data)
        if quant["signal_count"] >= 1 or quant["squeeze_ready"]:
            candidates.append({**data, "quant": quant})
    return candidates[:MAX_CANDIDATES]


# ══════════════════════════════════════════════════════════════════════════════
# TRADE LEVELS
# ══════════════════════════════════════════════════════════════════════════════

def calc_trade_levels(data: dict, direction: str, driver: str) -> dict:
    price      = data["price"]
    prev_close = data["prev_close"]
    gap_open   = prev_close * (1 + data["gap_pct"] / 100)
    atr        = data.get("atr", 0) or price * 0.05
    vwap       = data.get("vwap", price)
    vwap_p1    = data.get("vwap_plus1", 0)
    vwap_p2    = data.get("vwap_plus2", 0)
    vwap_m1    = data.get("vwap_minus1", 0)

    atr_mult   = 2.0 if driver in {"SQUEEZE","COMBINED"} else 1.5

    if direction == "LONG":
        stop_loss = max(round(price - atr * atr_mult, 2), round(gap_open * 0.99, 2))
    else:
        stop_loss = min(round(price + atr * atr_mult, 2), round(gap_open * 1.01, 2))

    risk = max(abs(price - stop_loss), 0.01)

    # TP1: 2:1 R/R
    tp1 = round(price + risk * 2.0, 2) if direction == "LONG" else round(price - risk * 2.0, 2)

    # TP2: VWAP band target (more accurate than just measured move)
    if direction == "LONG":
        tp2 = vwap_p2 if vwap_p2 > tp1 else round(price + abs(price - prev_close), 2)
        w52h = data.get("52w_high", 0)
        if w52h and price < w52h < tp2 * 1.5: tp2 = round(w52h * 0.99, 2)
    else:
        tp2 = vwap_m1 if vwap_m1 and vwap_m1 < tp1 else round(price - abs(price - prev_close), 2)
        w52l = data.get("52w_low", 0)
        if w52l and price > w52l > tp2 * 0.5: tp2 = round(w52l * 1.01, 2)

    tp2 = round(tp2, 2)
    tp1_rr = round(abs(tp1 - price) / risk, 2)
    tp2_rr = round(abs(tp2 - price) / risk, 2)

    if direction == "LONG":
        entry_ideal = round(min(vwap, gap_open) * 1.005, 2)
        entry_max   = round(price * 1.02, 2)
        entry_note  = (f"Enter ${entry_ideal}–${entry_max} — "
                       f"pullback to VWAP (${vwap}) or gap open (${round(gap_open,2)}), "
                       f"first green candle. TP1 at VWAP+1σ (${vwap_p1}), TP2 at VWAP+2σ (${vwap_p2})")
    else:
        entry_ideal = round(max(vwap, gap_open) * 0.995, 2)
        entry_max   = round(price * 0.98, 2)
        entry_note  = (f"Enter ${entry_max}–${entry_ideal} — "
                       f"bounce to VWAP (${vwap}) fails, first red candle. "
                       f"TP1 ${tp1}, TP2 at VWAP-1σ (${vwap_m1})")

    return {
        "gap_open": round(gap_open, 2), "stop_loss": stop_loss,
        "stop_type": f"ATR×{atr_mult} + structural",
        "risk_per_share": round(risk, 2),
        "tp1": tp1, "tp1_rr": tp1_rr, "tp1_action": "Sell 50% — move stop to breakeven",
        "tp2": tp2, "tp2_rr": tp2_rr, "tp2_action": "Sell 25% — trail final 25% at ATR×1.0",
        "trailing_stop_distance": round(atr, 2),
        "entry_ideal": entry_ideal, "entry_max": entry_max,
        "entry_note_computed": entry_note, "atr_mult_used": atr_mult,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI BRIEFING — enriched with all indicators
# ══════════════════════════════════════════════════════════════════════════════

def build_briefing(stock: dict) -> str:
    q          = stock["quant"]
    flags_txt  = "\n".join(f"  ✓ {f}" for f in q["flags"])    or "  None"
    conc_txt   = "\n".join(f"  ⚠ {c}" for c in q["concerns"]) or "  None"
    atr_pct    = stock.get("atr_pct", 0)
    patterns   = stock.get("patterns", {})
    pat_txt    = "\n".join(f"  ◆ {p}" for p in patterns.get("pattern_summary",[])) or "  None detected"
    orb        = stock.get("orb", {})
    vwap_bands = (f"  VWAP+1σ: ${stock.get('vwap_plus1',0)} | "
                  f"VWAP+2σ: ${stock.get('vwap_plus2',0)} | "
                  f"VWAP-1σ: ${stock.get('vwap_minus1',0)} | "
                  f"VWAP-2σ: ${stock.get('vwap_minus2',0)}")

    return f"""
MARKET INTELLIGENCE BRIEFING v4
Stock: {stock['name']} ({stock['ticker']})  |  Sector: {stock['sector']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}

━━━ PRICE & VOLUME ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Price: ${stock['price']}  |  Prev Close: ${stock['prev_close']}  |  Gap: {stock['gap_pct']:+.2f}%
  Volume: {stock['volume']:,} ({stock['volume_mult']:.1f}x 20d avg)
  ATR-14: ${stock['atr']} ({atr_pct:.1f}% of price)

━━━ TECHNICAL INDICATORS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VWAP: ${stock['vwap']}  |  Distance from VWAP: {stock['vwap_dist_pct']:+.1f}%
  VWAP Bands:
{vwap_bands}
  RSI-14: {stock['rsi']}  |  RSI Divergence: {stock['rsi_divergence']}
  ORB High: ${orb.get('orb_high',0)}  |  ORB Low: ${orb.get('orb_low',0)}  |  Status: {orb.get('orb_status','—')}
  RS vs SPY today: {stock['rs_vs_spy']:+.1f}%  (SPY: {stock['spy_change']:+.1f}%)

━━━ CANDLESTICK PATTERNS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{pat_txt}

━━━ SHORT INTEREST & FLOAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Short Interest: {stock['short_pct']:.1f}%  |  Trend: {stock.get('si_trend','—')}
  Float: {f"{stock['float_shares']/1e6:.1f}M" if stock['float_shares'] else 'Unknown'} shares

━━━ 52-WEEK RANGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  High: ${stock['52w_high']}  |  Low: ${stock['52w_low']}

━━━ QUANT FLAGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{flags_txt}

━━━ CONCERNS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conc_txt}

PEAD Candidate: {'YES' if q['pead_ready'] else 'NO'}  |  Squeeze Candidate: {'YES' if q['squeeze_ready'] else 'NO'}

━━━ YOUR TASK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are a seasoned professional trader. Use ALL indicators above.
RSI {stock['rsi']}, candlestick patterns, VWAP distance {stock['vwap_dist_pct']:+.1f}%, 
ORB {orb.get('orb_status','unknown')}, RS vs SPY {stock['rs_vs_spy']:+.1f}% — 
integrate these into your directional assessment.

Respond ONLY in exact JSON:
{{
  "direction": "LONG" | "SHORT" | "NO TRADE",
  "thesis": "2-3 sentences integrating technical + fundamental factors",
  "primary_driver": "PEAD" | "SQUEEZE" | "MOMENTUM" | "COMBINED" | "NONE",
  "timeframe": "e.g. 3-5 days",
  "key_risk": "the one condition that immediately kills this trade",
  "conviction": "HIGH" | "MODERATE" | "LOW",
  "entry_confirmation": "specific price action or pattern to see before entering",
  "entry_timing": "e.g. after 10:30am ET, on ORB breakout",
  "stop_reasoning": "why stop placement makes sense given ATR and structure",
  "exit_rule": "what signals thesis failure beyond the stop"
}}
"""

def call_claude(briefing: str) -> dict | None:
    if not ANTHROPIC_API_KEY:
        log.warning("No API key")
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 900,
                  "messages": [{"role": "user", "content": briefing}]},
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
        return json.loads(text)
    except Exception as e:
        log.error(f"Claude API: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PERSISTENCE LAYER
# ══════════════════════════════════════════════════════════════════════════════

HOLD_MS = {
    "COMBINED": 5*24*3600*1000, "PEAD": 5*24*3600*1000,
    "SQUEEZE":  int(2.5*24*3600*1000), "MOMENTUM": int(1.5*24*3600*1000), "NONE": 24*3600*1000,
}

def update_signal_store(ticker: str, signal: dict):
    """Increment seen count, promote to confirmed if threshold met."""
    now = int(time.time() * 1000)
    with SIGNAL_LOCK:
        if ticker not in SIGNAL_STORE:
            SIGNAL_STORE[ticker] = {"count": 0, "signal": None, "confirmed_at": None, "expires_at": None}
        entry = SIGNAL_STORE[ticker]
        entry["count"]      += 1
        entry["signal"]      = signal
        entry["last_seen"]   = now
        conviction = signal.get("conviction","LOW")
        if (entry["count"] >= CONFIRM_THRESHOLD and
            conviction in ("HIGH","MODERATE") and
            entry["confirmed_at"] is None):
            entry["confirmed_at"] = now
            driver = signal.get("primary_driver","NONE")
            entry["expires_at"] = now + HOLD_MS.get(driver, HOLD_MS["NONE"])
            log.info(f"★ CONFIRMED: {ticker} [{signal.get('direction')}] expires {datetime.fromtimestamp(entry['expires_at']/1000).strftime('%Y-%m-%d %H:%M')}")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def analyse_candidates(candidates: list) -> list:
    signals = []
    for stock in candidates:
        log.info(f"  → AI: {stock['ticker']}...")
        ai = call_claude(build_briefing(stock))
        if not ai or ai.get("direction") == "NO TRADE":
            log.info(f"    {stock['ticker']}: NO TRADE"); continue

        direction = ai["direction"]
        driver    = ai.get("primary_driver","NONE")
        levels    = calc_trade_levels(stock, direction, driver)

        # Build indicator payload for dashboard
        indicators = {
            "vwap":         stock["vwap"],
            "vwap_dist_pct":stock["vwap_dist_pct"],
            "vwap_plus1":   stock["vwap_plus1"],
            "vwap_plus2":   stock["vwap_plus2"],
            "vwap_minus1":  stock["vwap_minus1"],
            "vwap_minus2":  stock["vwap_minus2"],
            "rsi":          stock["rsi"],
            "rsi_divergence": stock["rsi_divergence"],
            "patterns":     stock["patterns"],
            "orb":          stock["orb"],
            "rs_vs_spy":    stock["rs_vs_spy"],
            "spy_change":   stock["spy_change"],
            "si_trend":     stock.get("si_trend",""),
        }

        # Partial signal dict for entry state eval
        partial_sig = {
            "direction":  direction, "price": stock["price"],
            "entry_ideal":levels["entry_ideal"], "entry_max":levels["entry_max"],
            "stop_loss":  levels["stop_loss"], "conviction": ai.get("conviction",""),
            "volume_mult":stock["volume_mult"],
        }
        # Lookup seen_count from store
        with SIGNAL_LOCK:
            seen = SIGNAL_STORE.get(stock["ticker"], {}).get("count", 0) + 1
        partial_sig["seen_count"] = seen
        entry_status = evaluate_entry_status(partial_sig, indicators)

        sig = {
            # Core
            "ticker":             stock["ticker"],
            "name":               stock["name"],
            "sector":             stock["sector"],
            "price":              stock["price"],
            "prev_close":         stock["prev_close"],
            "gap_pct":            stock["gap_pct"],
            "volume_mult":        stock["volume_mult"],
            "short_pct":          stock["short_pct"],
            "float_shares":       stock["float_shares"],
            "52w_high":           stock["52w_high"],
            "52w_low":            stock["52w_low"],
            # ATR + levels
            "atr":                stock["atr"],
            "atr_pct":            stock["atr_pct"],
            "vwap":               stock["vwap"],
            "gap_open":           levels["gap_open"],
            "stop_loss":          levels["stop_loss"],
            "stop_type":          levels["stop_type"],
            "risk_per_share":     levels["risk_per_share"],
            "tp1":                levels["tp1"], "tp1_rr": levels["tp1_rr"],
            "tp1_action":         levels["tp1_action"],
            "tp2":                levels["tp2"], "tp2_rr": levels["tp2_rr"],
            "tp2_action":         levels["tp2_action"],
            "trailing_stop":      levels["trailing_stop_distance"],
            "entry_ideal":        levels["entry_ideal"],
            "entry_max":          levels["entry_max"],
            "entry_note_computed":levels["entry_note_computed"],
            # Indicators
            **indicators,
            # Entry state
            "entry_status":       entry_status,
            "seen_count":         seen,
            # AI
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
            "source":             stock.get("source","watchlist"),
        }

        update_signal_store(sig["ticker"], sig)
        signals.append(sig)
        log.info(f"    {stock['ticker']}: {direction} | {ai.get('conviction')} | RSI:{stock['rsi']} | {entry_status['status']}")
    return signals


# ══════════════════════════════════════════════════════════════════════════════
# SCAN LOOP
# ══════════════════════════════════════════════════════════════════════════════

_batch_index = 0

def get_batch() -> list:
    global _batch_index
    total = max(1, len(BASE_WATCHLIST) // BATCH_SIZE)
    start = _batch_index * BATCH_SIZE
    batch = BASE_WATCHLIST[start:start + BATCH_SIZE]
    _batch_index = (_batch_index + 1) % total
    return batch

def run_scan():
    log.info("━━━ SCAN v4 STARTING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    get_spy_change()   # refresh SPY
    finviz = get_finviz_tickers()
    batch  = get_batch()
    fset   = set(finviz)
    universe = list(dict.fromkeys(finviz + batch))
    log.info(f"Universe: {len(universe)} | Finviz: {len(finviz)} | Batch: {len(batch)}")

    candidates = filter_candidates(universe)
    for c in candidates:
        c["source"] = "finviz" if c["ticker"] in fset else "watchlist"

    signals = analyse_candidates(candidates)

    # Build confirmed list from SIGNAL_STORE
    confirmed = []
    with SIGNAL_LOCK:
        for ticker, entry in SIGNAL_STORE.items():
            if entry.get("confirmed_at") and entry.get("signal"):
                s = dict(entry["signal"])
                s["confirmed_at"]  = entry["confirmed_at"]
                s["expires_at"]    = entry["expires_at"]
                s["seen_count"]    = entry["count"]
                s["last_seen"]     = entry.get("last_seen", 0)
                # Thesis status: is it still showing in signals?
                still_active = any(sig["ticker"] == ticker for sig in signals)
                if still_active:
                    s["thesis_status"] = "INTACT"
                else:
                    # Re-evaluate with latest data if possible
                    current_price = s.get("price", 0)
                    vwap          = s.get("vwap", 0)
                    s["thesis_status"] = evaluate_thesis_status(s, current_price, vwap)
                confirmed.append(s)

    with STATE_LOCK:
        STATE["signals"]      = [s for s in signals if not any(c["ticker"]==s["ticker"] for c in confirmed)]
        STATE["confirmed"]    = confirmed
        STATE["candidates"]   = [
            {"ticker":c["ticker"],"name":c["name"],"gap_pct":c["gap_pct"],
             "volume_mult":c["volume_mult"],"short_pct":c["short_pct"],
             "atr_pct":c.get("atr_pct",0),"rsi":c.get("rsi",50),
             "source":c.get("source","watchlist"),"quant_flags":c["quant"]["flags"]}
            for c in candidates
        ]
        STATE["watchlist"]    = BASE_WATCHLIST
        STATE["last_updated"] = datetime.now().isoformat()
        STATE["scan_count"]  += 1

    log.info(f"━━━ DONE: {len(signals)} signals | {len(confirmed)} confirmed ━━━━━━━━━━━━")

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

@app.route("/api/confirmed")
def api_confirmed():
    with STATE_LOCK:
        return jsonify(STATE.get("confirmed",[]))

@app.route("/api/health")
def api_health():
    with STATE_LOCK:
        return jsonify({"status":"online","scan_count":STATE["scan_count"],
                        "watchlist_size":len(BASE_WATCHLIST),"finviz_count":len(STATE.get("finviz_tickers",[])),
                        "confirmed_count":len(STATE.get("confirmed",[]))})

@app.route("/api/watchlist")
def api_watchlist():
    return jsonify({"tickers":BASE_WATCHLIST,"count":len(BASE_WATCHLIST)})

@app.route("/")
def index():
    with STATE_LOCK:
        sc = STATE["scan_count"]; cf = len(STATE.get("confirmed",[])); fv = len(STATE.get("finviz_tickers",[]))
    return (f"<h2>MIS Engine v4 ✓</h2>"
            f"<p>Watchlist: {len(BASE_WATCHLIST)} | Batch: {BATCH_SIZE} | Scans: {sc}</p>"
            f"<p>Confirmed: {cf} | Finviz: {fv} cached</p>")

if __name__ == "__main__":
    log.info(f"▶ MIS Engine v4 | {len(BASE_WATCHLIST)} tickers | Finviz: {FINVIZ_ENABLED}")
    threading.Thread(target=scan_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
