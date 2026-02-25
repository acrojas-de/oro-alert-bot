import os
import json
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

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.bfill().fillna(50)

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
        return {"meta": {}, "series": [], "signals": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"meta": {}, "series": [], "signals": []}

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

    # RSI Oro (para señales)
    gold_rsi = rsi(gold, 14)
    g_rsi14 = float(gold_rsi.iloc[i_g])
    g_rsi14_prev = float(gold_rsi.iloc[i_g - 1]) if len(gold) >= 3 else g_rsi14

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

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    row = {
        "ts": ts,
        "score": score,
        "bias": bias,
        "gold": round(g, 2),
        "gold_ema21": round(g_ema21, 2),
        "gold_ema50": round(g_ema50, 2),
        "gold_rsi14": round(g_rsi14, 2),
        "gold_rsi14_prev": round(g_rsi14_prev, 2),
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
    signals = data.get("signals", [])

    # Evita duplicado si se lanza dos veces rápido
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

    # ===== SEÑALES (BUY/SELL/WARN) =====
    if len(series) >= 2:
        prev = series[-2]
        curr = series[-1]

        p0 = prev["gold"]; p1 = curr["gold"]
        e210 = prev["gold_ema21"]; e211 = curr["gold_ema21"]
        e500 = prev["gold_ema50"]; e501 = curr["gold_ema50"]
        r0 = prev.get("gold_rsi14", 50); r1 = curr.get("gold_rsi14", 50)

        cross_up_ema21 = (p0 <= e210) and (p1 > e211)
        cross_dn_ema21 = (p0 >= e210) and (p1 < e211)

        ema_cross_up = (e210 <= e500) and (e211 > e501)
        ema_cross_dn = (e210 >= e500) and (e211 < e501)

        ema_bull = (e211 > e501)

        def add_signal(kind: str, reason: str, strength: int = 1):
            signals.append({
                "ts": curr["ts"],
                "asset": "GOLD",
                "type": kind,
                "reason": reason,
                "strength": int(strength)
            })

        if ema_cross_up:
            add_signal("BUY", "EMA21 cruza por encima de EMA50 (tendencia alcista)", 3)
        elif ema_cross_dn:
            add_signal("SELL", "EMA21 cruza por debajo de EMA50 (tendencia bajista)", 3)
        else:
            if cross_up_ema21 and ema_bull and r1 >= 50:
                add_signal("BUY", "Precio rompe EMA21 con tendencia a favor y RSI>50", 2)
            if cross_dn_ema21:
                add_signal("SELL", "Precio pierde EMA21 (posible debilidad)", 2)

        if r1 >= 70 and r1 < r0:
            add_signal("WARN", "RSI>70 y girando a la baja (agotamiento)", 1)
        elif r1 >= 70:
            add_signal("WARN", "RSI>=70 (sobrecompra)", 1)

        if r1 <= 30 and r1 > r0:
            add_signal("WARN", "RSI<30 y girando al alza (rebote)", 1)
        elif r1 <= 30:
            add_signal("WARN", "RSI<=30 (sobreventa)", 1)

    if len(signals) > 200:
        signals = signals[-200:]
    data["signals"] = signals

    save_data(DATA_PATH, data)
    print(f"Saved dashboard data: {DATA_PATH} (points={len(series)}) | signals={len(signals)}")

if __name__ == "__main__":
    main()
