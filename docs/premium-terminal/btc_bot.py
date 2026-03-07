import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# =========================================================
# ZONA 0 · CONFIGURACIÓN GENERAL
# Aquí defines activo, rutas y timeframes del maestro BTC
# =========================================================
BTC = "BTC-USD"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "btc_data.json"

TIMEFRAMES = {
    "4h": ("60d", "4h"),
    "1h": ("30d", "1h"),
    "5m": ("5d", "5m"),
    "1m": ("1d", "1m"),
}


# =========================================================
# ZONA 1 · INDICADORES BASE
# EMA y MACD: cimientos técnicos del radar
# =========================================================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


# =========================================================
# ZONA 2 · LIQUIDEZ ESTRUCTURAL
# Aquí sacamos zonas probables de liquidez arriba y abajo
# - swing_high / swing_low
# - range_high_20 / range_low_20
# Estas son las primeras referencias del "hotel"
# =========================================================
# =========================================================
# ZONA 2B · BARRIDAS DE LIQUIDEZ
# Detecta si el precio rompe una zona y vuelve dentro
# =========================================================
def detect_liquidity_sweep(close: pd.Series, liq: dict) -> dict:
    s = close.dropna()

    if len(s) < 2:
        return {
            "sweep_high": False,
            "sweep_low": False
        }

    prev_close = float(s.iloc[-2])
    last_close = float(s.iloc[-1])

    range_high = float(liq["range_high_20"])
    range_low = float(liq["range_low_20"])

    sweep_high = prev_close <= range_high and last_close > range_high
    sweep_low = prev_close >= range_low and last_close < range_low

    return {
        "sweep_high": sweep_high,
        "sweep_low": sweep_low
    }


# =========================================================
# ZONA 3 · FETCH POR TIMEFRAME
# Aquí se descarga el activo, se limpian datos,
# se calculan EMA, MACD y liquidez, y se devuelve
# todo listo para el JSON
# =========================================================
def fetch(tf: str) -> dict:
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
        raise RuntimeError(f"Sin datos para {BTC} en {tf}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower).dropna()

    if "close" not in df.columns:
        raise RuntimeError(f"No existe columna close para {BTC} en {tf}")

    close = df["close"].dropna()

    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    macd_line, signal, hist = macd(close)

    # -----------------------------
    # SUBZONA 3A · LIQUIDEZ DEL TF
    # -----------------------------
    liq = liquidity_levels(close)
   sweep = detect_liquidity_sweep(close, liq)

    # -----------------------------
    # SUBZONA 3B · SERIE DEL TF
    # -----------------------------
    out = []
    for i in range(len(close)):
        if pd.isna(close.iloc[i]) or pd.isna(ema21.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue

        out.append({
            "price": round(float(close.iloc[i]), 6),
            "ema21": round(float(ema21.iloc[i]), 6),
            "ema50": round(float(ema50.iloc[i]), 6),
            "macd": round(float(macd_line.iloc[i]), 6) if pd.notna(macd_line.iloc[i]) else None,
            "hist": round(float(hist.iloc[i]), 6) if pd.notna(hist.iloc[i]) else None
        })

   return {
    "series": out[-120:],
    "liquidity": liq,
    "sweep": sweep
}


# =========================================================
# ZONA 4 · CONSTRUCCIÓN DEL JSON FINAL
# Aquí se monta la salida que leerá btc.html
# - meta
# - signals
# - series
# - state
# =========================================================
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

    # -----------------------------------------------------
    # SUBZONA 4A · RECORRER TODOS LOS TIMEFRAMES
    # Aquí guardamos:
    # - data["series"][tf]
    # - data["state"][tf]["liquidity"]
    # -----------------------------------------------------
    for tf in TIMEFRAMES:
        tf_data = fetch(tf)

        data["series"][tf] = tf_data["series"]

       data["state"][tf] = {
    "liquidity": tf_data["liquidity"],
    "sweep": tf_data["sweep"]
}

    # -----------------------------------------------------
    # SUBZONA 4B · GUARDAR JSON EN DISCO
    # -----------------------------------------------------
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"BTC data updated: {DATA_PATH}")


# =========================================================
# ZONA 5 · ARRANQUE PRINCIPAL
# Punto de entrada del bot
# =========================================================
if __name__ == "__main__":
    main()
