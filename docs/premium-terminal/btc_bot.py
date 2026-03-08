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
# =========================================================
def liquidity_levels(close: pd.Series) -> dict:
    s = close.dropna()

    if len(s) < 20:
        last_price = float(s.iloc[-1]) if len(s) else 0.0
        return {
            "swing_high": round(last_price, 6),
            "swing_low": round(last_price, 6),
            "range_high_20": round(last_price, 6),
            "range_low_20": round(last_price, 6),
        }

    recent_20 = s.tail(20)
    recent_10 = s.tail(10)

    return {
        "swing_high": round(float(recent_10.max()), 6),
        "swing_low": round(float(recent_10.min()), 6),
        "range_high_20": round(float(recent_20.max()), 6),
        "range_low_20": round(float(recent_20.min()), 6),
    }


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
# ZONA 2C · DETECTOR DE COMPRESIÓN (ACUMULACIÓN)
# Busca mercado dormido + MACD comprimido
# =========================================================
def detect_compression(close: pd.Series, hist: pd.Series) -> dict:
    s_close = close.dropna()
    s_hist = hist.dropna()

    if len(s_close) < 20 or len(s_hist) < 10:
        return {
            "compression": False,
            "volatility": 0.0,
            "macd_compression": 0.0
        }

    volatility = s_close.pct_change().rolling(10).std().iloc[-1]
    hist_abs = s_hist.abs().rolling(10).mean().iloc[-1]

    volatility = 0.0 if pd.isna(volatility) else float(volatility)
    hist_abs = 0.0 if pd.isna(hist_abs) else float(hist_abs)

    compression = bool(volatility < 0.002 and hist_abs < 50)

    return {
        "compression": compression,
        "volatility": volatility,
        "macd_compression": hist_abs
    }


# =========================================================
# ZONA 2D · LIQUIDITY MAGNET
# Detecta hacia qué nivel de liquidez está más cerca el precio
# =========================================================
def detect_liquidity_magnet(close: pd.Series, liq: dict) -> dict:
    s = close.dropna()

    if len(s) == 0:
        return {
            "direction": "none",
            "target": 0.0,
            "distance_up": 0.0,
            "distance_down": 0.0
        }

    last_price = float(s.iloc[-1])

    up_target = float(liq.get("range_high_20", last_price))
    down_target = float(liq.get("range_low_20", last_price))

    distance_up = abs(up_target - last_price)
    distance_down = abs(last_price - down_target)

    if distance_up < distance_down:
        direction = "up"
        target = up_target
    elif distance_down < distance_up:
        direction = "down"
        target = down_target
    else:
        direction = "neutral"
        target = last_price

    return {
        "direction": direction,
        "target": round(float(target), 6),
        "distance_up": round(float(distance_up), 6),
        "distance_down": round(float(distance_down), 6)
    }


# =========================================================
# ZONA 2E · LIQUIDITY VACUUM
# Detecta zonas donde el precio puede moverse rápido
# =========================================================
def detect_liquidity_vacuum(close: pd.Series, liq: dict) -> dict:
    s = close.dropna()

    if len(s) < 10:
        return {
            "vacuum": False,
            "direction": "none",
            "target": 0.0,
            "distance_up": 0.0,
            "distance_down": 0.0
        }

    price = float(s.iloc[-1])

    high = float(liq.get("range_high_20", price))
    low = float(liq.get("range_low_20", price))

    dist_high = abs(high - price)
    dist_low = abs(price - low)

    vacuum_up = dist_high > (price * 0.015)
    vacuum_down = dist_low > (price * 0.015)

    if vacuum_down and dist_low > dist_high:
        return {
            "vacuum": True,
            "direction": "down",
            "target": round(low, 6),
            "distance_up": round(dist_high, 6),
            "distance_down": round(dist_low, 6)
        }

    if vacuum_up and dist_high > dist_low:
        return {
            "vacuum": True,
            "direction": "up",
            "target": round(high, 6),
            "distance_up": round(dist_high, 6),
            "distance_down": round(dist_low, 6)
        }

    return {
        "vacuum": False,
        "direction": "none",
        "target": round(price, 6),
        "distance_up": round(dist_high, 6),
        "distance_down": round(dist_low, 6)
    }


    
# =========================================================
# ZONA 2F · BIAS ENGINE
# Resume todas las señales en un sesgo de mercado
# =========================================================

TF_WEIGHTS = {
    "4h": 3.0,
    "1h": 2.0,
    "5m": 1.0,
    "1m": 0.5
}

def compute_bias(state: dict) -> dict:

    score_bull = 0
    score_bear = 0
    targets = []

    for tf, data in state.items():

        weight = TF_WEIGHTS.get(tf, 1)

        magnet = data.get("magnet", {})
        sweep = data.get("sweep", {})
        compression = data.get("compression", {})
        vacuum = data.get("vacuum", {})

        print("-----")
        print("TF:", tf)
        print("weight:", weight)
        print("magnet:", magnet)
        print("sweep:", sweep)
        print("compression:", compression)
        print("vacuum:", vacuum)

        # MAGNET
        if magnet.get("direction") == "up":
            score_bull += 30 * weight
            targets.append(magnet.get("target"))

        elif magnet.get("direction") == "down":
            score_bear += 30 * weight
            targets.append(magnet.get("target"))

        # SWEEP
        if sweep.get("sweep_high"):
            score_bear += 15 * weight

        if sweep.get("sweep_low"):
            score_bull += 15 * weight

        # COMPRESSION
        if compression.get("compression"):
            score_bull += 5 * weight
            score_bear += 5 * weight

        # VACUUM
        if vacuum.get("vacuum"):

            if vacuum.get("direction") == "up":
                score_bull += 20 * weight
                targets.append(vacuum.get("target"))

            elif vacuum.get("direction") == "down":
                score_bear += 20 * weight
                targets.append(vacuum.get("target"))

    total = score_bull + score_bear

    if total == 0:
        bull_pct = 50
        bear_pct = 50
    else:
        bull_pct = round(score_bull / total * 100, 2)
        bear_pct = round(score_bear / total * 100, 2)

    bias = "bullish" if bull_pct > bear_pct else "bearish"

    target = None
    if targets:
        target = round(sum(targets) / len(targets), 6)

    print("==== BIAS DEBUG ====")
    print("score_bull:", score_bull)
    print("score_bear:", score_bear)
    print("bull_pct:", bull_pct)
    print("bear_pct:", bear_pct)
    print("target:", target)

    return {
        "bias": bias,
        "bullish_pct": bull_pct,
        "bearish_pct": bear_pct,
        "target": target
    }


# =========================================================
# ZONA 3 · FETCH POR TIMEFRAME
# Aquí se descarga el activo, se limpian datos,
# se calculan EMA, MACD, liquidez, barridas y compresión
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
    # SUBZONA 3A · ESTADO DEL TF
    # -----------------------------
    liq = liquidity_levels(close)
    sweep = detect_liquidity_sweep(close, liq)
    compression = detect_compression(close, hist)
    magnet = detect_liquidity_magnet(close, liq)
    vacuum = detect_liquidity_vacuum(close, liq)

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
        "sweep": sweep,
        "compression": compression,
        "magnet": magnet,
        "vacuum": vacuum
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
    # -----------------------------------------------------
    for tf in TIMEFRAMES:
        tf_data = fetch(tf)

        data["series"][tf] = tf_data["series"]

        data["state"][tf] = {
            "liquidity": tf_data["liquidity"],
            "sweep": tf_data["sweep"],
            "compression": tf_data["compression"],
            "magnet": tf_data["magnet"],
            "vacuum": tf_data["vacuum"]
        }

    data["bias"] = compute_bias(data["state"])
    
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
