"""
Daily NSE 3% Momentum Stock Scanner
====================================
Run:  python scanner.py
Output: results_YYYYMMDD.csv  (top 50 ranked stocks)

Dependencies (already installed):
    pip install pandas numpy yfinance pandas_ta_classic nselib requests openpyxl
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Silence noisy loggers
# ---------------------------------------------------------------------------
for _n in ("yfinance", "peewee", "urllib3"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TOP_N            = 50          # stocks to output
MIN_AVG_VALUE    = 5_00_00_000 # min 5 Cr avg daily traded value (liquidity filter)
MIN_PRICE        = 20          # min price in Rs
MAX_PRICE        = 50_000      # max price
HISTORY_YEARS    = 1           # reduced: 1yr of daily history
INTRADAY_DAYS    = 59          # yfinance caps 1h history at ~60 days
MAX_WORKERS      = 10          # reduced threads
BATCH_SIZE       = 50

ROOT    = Path(__file__).parent
DT      = datetime.today().strftime("%Y%m%d")
OUT_CSV = ROOT / f"results_{DT}.csv"
OUT_XLS = ROOT / f"results_{DT}.xlsx"

# ---------------------------------------------------------------------------
# STEP 1: Get NSE universe
# ---------------------------------------------------------------------------
def get_universe() -> list[str]:
    """Return Nifty 500 tickers only — lightweight universe."""
    print("Fetching Nifty 500 universe...")
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        df = pd.read_csv(url)
        symbols = (df["Symbol"] + ".NS").tolist()
        print(f"  {len(symbols)} symbols loaded")
        return symbols
    except Exception as e:
        print(f"  NSE fetch failed ({e}), using Nifty 50 fallback")
        nifty50 = [
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC",
            "SBIN","BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
            "SUNPHARMA","TITAN","BAJFINANCE","NESTLEIND","WIPRO","ULTRACEMCO",
            "POWERGRID","NTPC","ONGC","TECHM","ADANIENT","HCLTECH","JSWSTEEL",
            "TATAMOTORS","TATASTEEL","M&M","INDUSINDBK","BAJAJ-AUTO","GRASIM",
            "DRREDDY","CIPLA","DIVISLAB","APOLLOHOSP","EICHERMOT","HEROMOTOCO",
            "BPCL","COALINDIA","TATACONSUM","HINDALCO","ADANIPORTS","SBILIFE",
            "BRITANNIA","HDFCLIFE","UPL","SHREECEM","BAJAJFINSV",
        ]
        return [s + ".NS" for s in nifty50]


# ---------------------------------------------------------------------------
# STEP 2: Batch download OHLCV
# ---------------------------------------------------------------------------
def download_batch(tickers: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Download OHLCV for a batch; returns {ticker: df}."""
    out: dict[str, pd.DataFrame] = {}
    ns_tickers = tickers  # already in .NS format
    try:
        raw = yf.download(
            " ".join(ns_tickers),
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            raise ValueError("empty")
        if len(tickers) == 1:
            if not raw.empty:
                out[tickers[0]] = raw
        else:
            for t in tickers:
                try:
                    df = raw[t].dropna(how="all")
                    if not df.empty:
                        out[t] = df
                except KeyError:
                    pass
    except Exception:
        # per-ticker fallback
        for t in tickers:
            try:
                df = yf.Ticker(t).history(period=period, interval=interval)
                if df is not None and not df.empty:
                    out[t] = df
            except Exception:
                pass
    return out


def download_all(tickers: list[str], period: str = "6mo", interval: str = "1d") -> dict[str, pd.DataFrame]:
    batches = [tickers[i: i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    result: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_batch, b, period, interval): b for b in batches}
        for fut in as_completed(futures):
            result.update(fut.result())
            done += len(futures[fut])
            pct = int(done * 100 / total)
            print(f"  Downloaded {done}/{total} ({pct}%)", end="\r")
    print()
    return result


# ---------------------------------------------------------------------------
# STEP 3: Technical indicators
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> dict | None:
    """Compute all TA indicators on daily OHLCV. Returns dict of latest values."""
    if df is None or len(df) < 50:
        return None
    try:
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        # EMAs
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(com=13, adjust=False).mean()

        # MACD
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Volume (relative)
        avg_vol_20 = volume.rolling(20).mean()
        rel_vol    = volume / avg_vol_20.replace(0, np.nan)

        # Average traded value
        avg_value_20 = (close * volume).rolling(20).mean()

        # Volatility
        log_ret   = np.log(close / close.shift())
        vol_5d    = log_ret.rolling(5).std()  * np.sqrt(252) * 100
        vol_20d   = log_ret.rolling(20).std() * np.sqrt(252) * 100

        # ADX (simplified)
        adx = _adx(high, low, close)

        # VWAP proxy (daily: use typical price ratio to 20d avg)
        typical = (high + low + close) / 3
        vwap_proxy = (typical * volume).rolling(20).sum() / volume.rolling(20).sum()

        # Previous day values
        prev_high  = high.iloc[-2]
        prev_low   = low.iloc[-2]
        prev_close = close.iloc[-2]

        # Gap
        gap_pct = (close.iloc[-1] - prev_close) / prev_close * 100

        c   = close.iloc[-1]
        atr_pct = atr.iloc[-1] / c * 100 if c > 0 else 0

        return {
            "close":        round(c, 2),
            "prev_close":   round(prev_close, 2),
            "prev_high":    round(prev_high, 2),
            "prev_low":     round(prev_low, 2),
            "ema9":         round(ema9.iloc[-1], 2),
            "ema20":        round(ema20.iloc[-1], 2),
            "ema50":        round(ema50.iloc[-1], 2),
            "rsi":          round(rsi.iloc[-1], 1),
            "atr":          round(atr.iloc[-1], 2),
            "atr_pct":      round(atr_pct, 2),
            "macd":         round(macd.iloc[-1], 4),
            "macd_signal":  round(signal.iloc[-1], 4),
            "adx":          round(adx.iloc[-1], 1) if adx is not None else None,
            "rel_vol":      round(rel_vol.iloc[-1], 2),
            "avg_value_20": round(avg_value_20.iloc[-1], 0),
            "vol_5d":       round(vol_5d.iloc[-1], 2),
            "vol_20d":      round(vol_20d.iloc[-1], 2),
            "vwap":         round(vwap_proxy.iloc[-1], 2),
            "gap_pct":      round(gap_pct, 2),
            "above_prev_high": int(c > prev_high),
            "pct_change":   round((c - prev_close) / prev_close * 100, 2),
        }
    except Exception:
        return None


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series | None:
    try:
        up   = high.diff()
        down = -low.diff()
        plus_dm  = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr      = tr.ewm(com=period - 1, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr
        dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(com=period - 1, adjust=False).mean()
        return adx
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STEP 4: Historical probability engine (daily data proxy)
# ---------------------------------------------------------------------------
def historical_probability(df: pd.DataFrame) -> dict:
    """
    Using daily OHLCV as proxy (true intraday 5-min not always available via free APIs).
    Estimates probability of +3%, +2%, +1.5% move in a day based on historical daily ranges.
    Returns probabilities and stats.
    """
    defaults = {
        "prob_3pct": 0.0,
        "prob_2pct": 0.0,
        "prob_1_5pct": 0.0,
        "hist_win_rate": 0.0,
        "avg_daily_range_pct": 0.0,
        "sample_size": 0,
    }
    try:
        if df is None or len(df) < 60:
            return defaults

        c     = df["Close"]
        h     = df["High"]
        l     = df["Low"]
        o     = df["Open"]

        # Daily range from open (available intraday proxy)
        up_move   = (h - o) / o * 100   # max upside from open
        down_move = (o - l) / o * 100   # max downside from open

        # Only use last 2 years
        lookback = min(len(df), HISTORY_YEARS * 252)
        up   = up_move.iloc[-lookback:]
        down = down_move.iloc[-lookback:]

        n = len(up)
        if n < 30:
            return defaults

        # Prob of reaching +X% before hitting -1%
        # Approximate: days where up_move >= X% AND down_move < 1%
        prob_3   = ((up >= 3.0) & (down < 1.0)).sum() / n
        prob_2   = ((up >= 2.0) & (down < 1.0)).sum() / n
        prob_1_5 = ((up >= 1.5) & (down < 1.0)).sum() / n

        # Historical win rate: days with positive close vs open
        win_rate = ((c - o) > 0).iloc[-lookback:].sum() / n

        avg_range = ((h - l) / l * 100).iloc[-lookback:].mean()

        return {
            "prob_3pct":          round(float(prob_3)   * 100, 1),
            "prob_2pct":          round(float(prob_2)   * 100, 1),
            "prob_1_5pct":        round(float(prob_1_5) * 100, 1),
            "hist_win_rate":      round(float(win_rate) * 100, 1),
            "avg_daily_range_pct":round(float(avg_range), 2),
            "sample_size":        n,
        }
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# STEP 5: Scoring (out of 100)
# ---------------------------------------------------------------------------
def score_stock(ind: dict, prob: dict) -> float:
    """
    Score out of 100 based on spec weights.
    """
    score = 0.0

    # 1. Historical probability of +3% (30 pts)
    # Scale: 0% prob = 0 pts, 50% prob = 30 pts (cap at 50%)
    score += min(prob.get("prob_3pct", 0) / 50 * 30, 30)

    # 2. Relative volume (15 pts)
    # Scale: rel_vol 1.0 = 0 pts, 3.0+ = 15 pts
    rv = ind.get("rel_vol", 1.0)
    score += min((rv - 1.0) / 2.0 * 15, 15)

    # 3. VWAP strength (15 pts)
    # Price above VWAP proxy = 15 pts, below = scale by proximity
    c    = ind.get("close", 0)
    vwap = ind.get("vwap", c)
    if vwap > 0:
        vwap_ratio = c / vwap
        score += min(max((vwap_ratio - 0.97) / 0.06 * 15, 0), 15)

    # 4. EMA alignment (10 pts)
    ema9  = ind.get("ema9",  0)
    ema20 = ind.get("ema20", 0)
    ema50 = ind.get("ema50", 0)
    ema_pts = 0
    if ema9  > ema20: ema_pts += 5
    if ema20 > ema50: ema_pts += 5
    score += ema_pts

    # 5. RSI + momentum (10 pts)
    rsi = ind.get("rsi", 50)
    if 55 <= rsi <= 75:
        score += 10
    elif 50 <= rsi < 55 or 75 < rsi <= 80:
        score += 5

    # 6. Breakout strength (10 pts)
    score += ind.get("above_prev_high", 0) * 10

    # 7. Liquidity (5 pts)
    avg_val = ind.get("avg_value_20", 0)
    score += min(avg_val / (5 * MIN_AVG_VALUE) * 5, 5)

    # 8. ATR suitability (5 pts)
    # ATR% >= 3% means the stock CAN make a 3% move
    atr_pct = ind.get("atr_pct", 0)
    if atr_pct >= 3.0:
        score += 5
    elif atr_pct >= 2.0:
        score += 3
    elif atr_pct >= 1.5:
        score += 1

    return round(min(score, 100), 1)


# ---------------------------------------------------------------------------
# STEP 6: Filters
# ---------------------------------------------------------------------------
def passes_filters(ticker: str, ind: dict) -> bool:
    c = ind.get("close", 0)
    if c < MIN_PRICE or c > MAX_PRICE:
        return False
    if ind.get("avg_value_20", 0) < MIN_AVG_VALUE:
        return False
    if ind.get("rel_vol", 0) < 0.5:
        return False
    return True


def build_reason(ind: dict, prob: dict) -> str:
    reasons = []
    if ind.get("close", 0) > ind.get("vwap", 0):
        reasons.append("Above VWAP")
    if ind.get("ema9", 0) > ind.get("ema20", 0) > ind.get("ema50", 0):
        reasons.append("Bullish EMA stack")
    rsi = ind.get("rsi", 0)
    if 55 <= rsi <= 75:
        reasons.append(f"RSI {rsi}")
    if ind.get("rel_vol", 0) >= 1.5:
        reasons.append(f"RelVol {ind['rel_vol']}x")
    if ind.get("above_prev_high", 0):
        reasons.append("Above prev high")
    if prob.get("prob_3pct", 0) >= 20:
        reasons.append(f"Hist3%={prob['prob_3pct']}%")
    return "; ".join(reasons) if reasons else "Momentum"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*60}")
    print("  Daily NSE 3% Momentum Stock Scanner")
    print(f"  Date: {datetime.today().strftime('%d-%b-%Y %H:%M')}")
    print(f"{'='*60}\n")

    # 1. Universe
    tickers = get_universe()

    # 2. Download daily OHLCV — 1yr only
    print("\nDownloading 1yr daily data...")
    hist_data = download_all(tickers, period="1y", interval="1d")
    print(f"  Got data for {len(hist_data)} stocks")

    # 3. Compute indicators + probabilities + score
    print("\nComputing indicators and scoring...")
    rows = []
    for ticker, df in hist_data.items():
        ind  = compute_indicators(df)
        if ind is None:
            continue
        if not passes_filters(ticker, ind):
            continue
        prob = historical_probability(df)
        sc   = score_stock(ind, prob)
        sym  = ticker.replace(".NS", "")
        entry  = ind["close"]
        target = round(entry * 1.03, 2)
        sl     = round(entry * 0.99, 2)
        rows.append({
            "Rank":               0,
            "Symbol":             sym,
            "Price":              ind["close"],
            "Change%":            ind["pct_change"],
            "VWAP":               ind["vwap"],
            "EMA9":               ind["ema9"],
            "EMA20":              ind["ema20"],
            "EMA50":              ind["ema50"],
            "RSI":                ind["rsi"],
            "RelVol":             ind["rel_vol"],
            "ATR%":               ind["atr_pct"],
            "Hist_Prob_3%":       prob["prob_3pct"],
            "Hist_Prob_2%":       prob["prob_2pct"],
            "Hist_Prob_1.5%":     prob["prob_1_5pct"],
            "Hist_WinRate%":      prob["hist_win_rate"],
            "AvgDailyRange%":     prob["avg_daily_range_pct"],
            "SampleSize":         prob["sample_size"],
            "TechScore":          sc,
            "Entry":              entry,
            "Target(+3%)":        target,
            "StopLoss(-1%)":      sl,
            "Reason":             build_reason(ind, prob),
        })

    if not rows:
        print("No stocks passed filters today.")
        return

    # 4. Rank
    df_out = pd.DataFrame(rows)
    df_out.sort_values("TechScore", ascending=False, inplace=True)
    df_out = df_out.head(TOP_N).reset_index(drop=True)
    df_out["Rank"] = df_out.index + 1

    # 5. Trade status
    def trade_status(row):
        if (row["Price"] > row["VWAP"] and
            row["EMA9"]  > row["EMA20"] and
            row["RelVol"] >= 1.5 and
            row["RSI"] >= 55 and
            row["Hist_Prob_3%"] >= 20):
            return "BUY CANDIDATE"
        return "WATCH"

    df_out["TradeStatus"] = df_out.apply(trade_status, axis=1)

    # 6. Best 5 and primary
    top5 = df_out[df_out["TradeStatus"] == "BUY CANDIDATE"].head(5)
    primary = top5.iloc[0]["Symbol"] if not top5.empty else "NO TRADE"

    # 7. Save
    col_order = [
        "Rank", "Symbol", "Price", "Change%", "VWAP", "EMA9", "EMA20", "EMA50",
        "RSI", "RelVol", "ATR%", "Hist_Prob_3%", "Hist_Prob_2%", "Hist_Prob_1.5%",
        "Hist_WinRate%", "AvgDailyRange%", "SampleSize", "TechScore",
        "Entry", "Target(+3%)", "StopLoss(-1%)", "TradeStatus", "Reason",
    ]
    df_out = df_out[col_order]
    df_out.to_csv(OUT_CSV, index=False)

    # 8. Print summary
    print(f"\n{'='*60}")
    print(f"  TOP {len(df_out)} STOCKS  |  {datetime.today().strftime('%d-%b-%Y')}")
    print(f"{'='*60}")
    pd.set_option("display.max_rows", TOP_N)
    pd.set_option("display.width", 200)
    print(df_out[["Rank","Symbol","Price","Change%","RSI","RelVol",
                  "Hist_Prob_3%","TechScore","TradeStatus"]].to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  PRIMARY BUY CANDIDATE : {primary}")
    if not top5.empty:
        print(f"  TOP 5 CANDIDATES      : {', '.join(top5['Symbol'].tolist())}")
    print(f"{'='*60}")
    print(f"\n  Saved: {OUT_CSV}")
    print(f"  Saved: {OUT_XLS}\n")


def git_sync():
    """Pull latest changes before run, push results after."""
    import subprocess
    cwd = str(ROOT)
    print("Git pull...")
    subprocess.run(["git", "pull"], cwd=cwd, check=False)

def git_push():
    import subprocess
    cwd = str(ROOT)
    print("Git push...")
    subprocess.run(["git", "add", "."],                             cwd=cwd, check=False)
    subprocess.run(["git", "commit", "-m", f"Results {DT}"],        cwd=cwd, check=False)
    subprocess.run(["git", "push"],                                 cwd=cwd, check=False)

if __name__ == "__main__":
    git_sync()
    main()
    git_push()
