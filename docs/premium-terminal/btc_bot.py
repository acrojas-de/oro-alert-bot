import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

BTC = "BTC-USD"

DATA_PATH = "docs/premium-terminal/btc_data.json"

TIMEFRAMES = {
    "4h": ("60d","4h"),
    "1h": ("30d","1h"),
    "5m": ("5d","5m"),
    "1m": ("1d","1m")
}

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def macd(series):
    ema12 = ema(series,12)
    ema26 = ema(series,26)
    macd_line = ema12 - ema26
    signal = ema(macd_line,9)
    hist = macd_line - signal
    return macd_line, signal, hist


def fetch(tf):
    period, interval = TIMEFRAMES[tf]

    df = yf.download(
        BTC,
        period=period,
        interval=interval,
        progress=False
    )

    df = df.rename(columns=str.lower)

    close = df["close"]

    ema21 = ema(close,21)
    ema50 = ema(close,50)

    macd_line, signal, hist = macd(close)

    out = []

    for i in range(len(close)):
        out.append({
            "price": float(close.iloc[i]),
            "ema21": float(ema21.iloc[i]),
            "ema50": float(ema50.iloc[i]),
            "macd": float(macd_line.iloc[i]),
            "hist": float(hist.iloc[i])
        })

    return out[-120:]


def main():

    data = {
        "meta":{
            "asset":"BTC-USD",
            "updated_utc": datetime.now(timezone.utc).isoformat()
        },
        "series":{},
        "signals":[]
    }

    for tf in TIMEFRAMES:
        data["series"][tf] = fetch(tf)

    os.makedirs(os.path.dirname(DATA_PATH),exist_ok=True)

    with open(DATA_PATH,"w") as f:
        json.dump(data,f,indent=2)

    print("BTC data updated")


if __name__ == "__main__":
    main()
