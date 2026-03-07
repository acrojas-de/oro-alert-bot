import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

BTC = "BTC-USD"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "btc_data.json"

TIMEFRAMES = {
    "4h": ("60d", "4h"),
    "1h": ("30d", "1h"),
    "5m": ("5d", "5m"),
    "1m": ("1d", "1m"),
}

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def fetch(tf):
    period, interval = TIMEFRAMES[tf]

    df = yf.download(
        BTC,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"Sin datos para {BTC} {tf}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower).dropna()

    if "close" not in df.columns:
        raise RuntimeError(f"No existe columna close para {BTC} {tf}")

    close = df["close"].dropna()

    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    macd_line, signal, hist = macd(close)

    out = []
    for i in range(len(close)):
        if pd.isna(close.iloc[i]) or pd.isna(ema21.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue

        out.append({
            "price": round(float(close.iloc[i]), 6),
            "ema21": round(float(ema21.iloc[i]), 6),
            "ema50": round(float(ema50.iloc[i]), 6),
            "macd": round(float(macd_line.iloc[i]), 6) if pd.notna(macd_line.iloc[i]) else None,
            "hist": round(float(hist.iloc[i]), 6) if pd.notna(hist.iloc[i]) else None,
        })

    return out[-120:]

def main():
    data = {
        "meta": {
            "asset": "BTC-USD",
            "updated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        },
        "signals": [],
        "series": {},
        "state": {}
    }

    for tf in TIMEFRAMES:
        data["series"][tf] = fetch(tf)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"BTC data updated: {DATA_PATH}")

if __name__ == "__main__":
    main()
