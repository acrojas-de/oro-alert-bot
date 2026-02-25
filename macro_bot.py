import os
import json
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# ===== CONFIG =====
TIMEFRAME = os.getenv("TIMEFRAME", "60m")
PERIOD = os.getenv("PERIOD", "30d")

GOLD_TICKER = os.getenv("GOLD_TICKER", "GC=F")
DXY_TICKER = os.getenv("DXY_TICKER", "DX-Y.NYB")
TNX_TICKER = os.getenv("TNX_TICKER", "^TNX")
NASDAQ_TICKER = os.getenv("NASDAQ_TICKER", "^IXIC")

# Dashboard data file (served by GitHub Pages)
DATA_PATH = os.getenv("MACRO_DATA_PATH", "docs/macro_data.json")

# Keep last N points
MAX_POINTS = int(os.getenv("MAX_POINTS", "500"))

# ===== HELPERS =====
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def fetch_close(ticker: str) -> pd.Series:
    df = yf.download(ticker, period=PERIOD, interval=TIMEFRAME, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Sin datos para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).dropna()
    if "close" not in df.columns:
        raise RuntimeError(f"Falta close en {ticker}. Columnas: {list(df.columns)}")
    return df["close"].dropna()

def last_closed_index(series: pd.Series) -> int:
    # último cierre confirmado (evita vela en formación)
    return -2 if len(series) >= 3 else -1

def load_data(path: str) -> dict:
    if not os.path.exists(path):
        return {"meta": {}, "series": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"meta": {}, "series": []}

def save_data(path: str, data: dict) -> None:
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ===== MAIN =====
def main():
    gold = fetch_close(GOLD_TICKER)
    dxy = fetch_close(DXY_TICKER)
    tnx = fetch_close(TNX_TICKER)
    nas = fetch_close(NASDAQ_TICKER)

    i_g = last_closed_index(gold)
    i_d = last_closed_index(dxy)
    i_t = last_closed_index(tnx)
    i_n = last_closed_index(nas)

    g = float(gold.iloc[i_g]); g_ema21 = float(ema(gold, 21).iloc[i_g]); g_ema50 = float(ema(gold, 50).iloc[i_g])
    d = float(dxy.iloc[i_d]);  d_ema21 = float(ema(dxy, 21).iloc[i_d])
    t = float(tnx.iloc[i_t]);  t_ema21 = float(ema(tnx, 21).iloc[i_t])
    n = float(nas.iloc[i_n]);  n_ema21 = float(ema(nas, 21).iloc[i_n])

    # ===== MACRO SCORE (0..100) =====
    score = 0
    score += 25 if (g > g_ema21 and g_ema21 > g_ema50) else 0
    score += 25 if (d < d_ema21) else 0
    score += 25 if (t < t_ema21) else 0
    score += 25 if (n < n_ema21) else 0

    if score >= 75:
        bias = "LONG_FAVORABLE"
    elif score <= 25:
        bias = "SHORT_CUIDADO"
    else:
        bias = "MIXTO_ESPERAR"

    # Timestamp (UTC ISO). En el dashboard lo formateamos.
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    row = {
        "ts": ts,
        "score": score,
        "bias": bias,
        "gold": round(g, 2),
        "gold_ema21": round(g_ema21, 2),
        "gold_ema50": round(g_ema50, 2),
        "dxy": round(d, 4),
        "dxy_ema21": round(d_ema21, 4),
        "tnx": round(t, 4),
        "tnx_ema21": round(t_ema21, 4),
        "nasdaq": round(n, 2),
        "nasdaq_ema21": round(n_ema21, 2),
        "tickers": {
            "gold": GOLD_TICKER,
            "dxy": DXY_TICKER,
            "tnx": TNX_TICKER,
            "nasdaq": NASDAQ_TICKER
        }
    }

    data = load_data(DATA_PATH)
    series = data.get("series", [])

    # Evita duplicado si el workflow se lanza dos veces rápido
    if series and series[-1].get("ts") == ts:
        print("No new point (same timestamp).")
        return

    series.append(row)
    if len(series) > MAX_POINTS:
        series = series[-MAX_POINTS:]

    data["meta"] = {
        "timeframe": TIMEFRAME,
        "period": PERIOD,
        "updated_utc": ts
    }
    data["series"] = series

    save_data(DATA_PATH, data)
    print(f"Saved dashboard data: {DATA_PATH} (points={len(series)})")

if __name__ == "__main__":
    main()
