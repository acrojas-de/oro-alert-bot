import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# =========================================================
# ZONA 0 · CONFIGURACIÓN GENERAL
# =========================================================
BTC = "BTC-USD"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "btc_data.json"

# Para 4H usaremos 1H y luego reagrupamos a 4H
TIMEFRAMES = {
    "4h": ("60d", "1h"),
    "1h": ("30d", "1h"),
    "5m": ("5d", "5m"),
    "1m": ("1d", "1m"),
}

TF_WEIGHTS = {
    "4h": 3.0,
    "1h": 2.0,
    "5m": 1.0,
    "1m": 0.5
}

# Telegram opcional
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "0") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# =========================================================
# ZONA 1 · INDICADORES BASE
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
# ZONA 2C · DETECTOR DE COMPRESIÓN
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
# ZONA 2G · TRAP DETECTOR
# =========================================================
def detect_trap(sweep: dict, magnet: dict, compression: dict) -> dict:
    sweep_high = bool(sweep.get("sweep_high", False))
    sweep_low = bool(sweep.get("sweep_low", False))
    magnet_dir = magnet.get("direction", "none")
    is_compression = bool(compression.get("compression", False))

    bull_trap = sweep_high and magnet_dir == "down"
    bear_trap = sweep_low and magnet_dir == "up"

    trap_type = "none"
    direction = "none"
    confidence = "low"

    if bull_trap:
        trap_type = "bull_trap"
        direction = "down"
        confidence = "high" if is_compression else "medium"
    elif bear_trap:
        trap_type = "bear_trap"
        direction = "up"
        confidence = "high" if is_compression else "medium"

    return {
        "trap": trap_type != "none",
        "type": trap_type,
        "direction": direction,
        "confidence": confidence,
        "compression_confirmed": is_compression
    }


# =========================================================
# ZONA 2F · BIAS ENGINE + TARGET DIRECCIONAL
# =========================================================
def compute_bias(state: dict) -> dict:
    score_bull = 0
    score_bear = 0

    targets_up = []
    targets_down = []

    for tf, data in state.items():
        weight = TF_WEIGHTS.get(tf, 1)

        magnet = data.get("magnet", {})
        sweep = data.get("sweep", {})
        compression = data.get("compression", {})
        vacuum = data.get("vacuum", {})
        trap = data.get("trap", {})

        if magnet.get("direction") == "up":
            score_bull += 30 * weight
            if magnet.get("target") is not None:
                targets_up.append((magnet.get("target"), weight))

        elif magnet.get("direction") == "down":
            score_bear += 30 * weight
            if magnet.get("target") is not None:
                targets_down.append((magnet.get("target"), weight))

        if sweep.get("sweep_high"):
            score_bear += 15 * weight

        if sweep.get("sweep_low"):
            score_bull += 15 * weight

        if compression.get("compression"):
            score_bull += 5 * weight
            score_bear += 5 * weight

        if vacuum.get("vacuum"):
            if vacuum.get("direction") == "up":
                score_bull += 20 * weight
                if vacuum.get("target") is not None:
                    targets_up.append((vacuum.get("target"), weight))

            elif vacuum.get("direction") == "down":
                score_bear += 20 * weight
                if vacuum.get("target") is not None:
                    targets_down.append((vacuum.get("target"), weight))

        if trap.get("trap"):
            if trap.get("direction") == "up":
                score_bull += 25 * weight
            elif trap.get("direction") == "down":
                score_bear += 25 * weight

    total = score_bull + score_bear

    if total == 0:
        bull_pct = 50.0
        bear_pct = 50.0
        bias = "neutral"
    else:
        bull_pct = round(score_bull / total * 100, 2)
        bear_pct = round(score_bear / total * 100, 2)
        if bull_pct > bear_pct:
            bias = "bullish"
        elif bear_pct > bull_pct:
            bias = "bearish"
        else:
            bias = "neutral"

    def weighted_avg(targets):
        if not targets:
            return None
        valid = [(t, w) for t, w in targets if t is not None]
        if not valid:
            return None
        total_w = sum(w for _, w in valid)
        if total_w == 0:
            return None
        return round(sum(t * w for t, w in valid) / total_w, 6)

    if bias == "bullish":
        target = weighted_avg(targets_up)
    elif bias == "bearish":
        target = weighted_avg(targets_down)
    else:
        target = None

    return {
        "bias": bias,
        "bullish_pct": bull_pct,
        "bearish_pct": bear_pct,
        "target": target
    }


# =========================================================
# ZONA 3A · DESCARGA Y NORMALIZACIÓN
# =========================================================
def download_raw(period: str, interval: str) -> pd.DataFrame | None:
    df = yf.download(
        BTC,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if df is None or df.empty:
        print(f"[WARN] Sin datos para {BTC} en intervalo {interval}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower).dropna(how="all")

    if "close" not in df.columns:
        print(f"[WARN] No existe columna close para {BTC} en intervalo {interval}")
        return None

    return df


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        print("[WARN] El índice no es DatetimeIndex para reagrupar a 4H")
        return None

    cols_needed = ["open", "high", "low", "close"]
    for c in cols_needed:
        if c not in df.columns:
            print(f"[WARN] Falta columna {c} para reagrupar a 4H")
            return None

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample("4h").agg(agg).dropna()

    if out.empty:
        print("[WARN] El resample a 4H devolvió vacío")
        return None

    return out


# =========================================================
# ZONA 3 · FETCH POR TIMEFRAME
# =========================================================
def fetch(tf: str) -> dict | None:
    period, interval = TIMEFRAMES[tf]

    raw = download_raw(period, interval)
    if raw is None:
        return None

    if tf == "4h":
        df = resample_to_4h(raw)
    else:
        df = raw.copy()

    if df is None or df.empty:
        print(f"[WARN] DataFrame vacío para {BTC} en {tf}")
        return None

    df = df.dropna()

    if "close" not in df.columns:
        print(f"[WARN] El dataframe de {tf} no contiene 'close'")
        return None

    close = df["close"].dropna()

    if len(close) < 30:
        print(f"[WARN] Datos insuficientes para {BTC} en {tf}: {len(close)} velas")
        return None

    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    macd_line, signal, hist = macd(close)

    liq = liquidity_levels(close)
    sweep = detect_liquidity_sweep(close, liq)
    compression = detect_compression(close, hist)
    magnet = detect_liquidity_magnet(close, liq)
    vacuum = detect_liquidity_vacuum(close, liq)
    trap = detect_trap(sweep, magnet, compression)

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
        "vacuum": vacuum,
        "trap": trap
    }


# =========================================================
# ZONA 4 · TELEGRAM INTELIGENTE (PREPARADO)
# =========================================================
def build_telegram_message(data: dict) -> str | None:
    bias = data.get("bias", {})
    state = data.get("state", {})

    if not bias:
        return None

    bias_side = bias.get("bias", "neutral")
    bull = bias.get("bullish_pct")
    bear = bias.get("bearish_pct")
    target = bias.get("target")

    if bias_side == "bullish" and (bull is None or bull < 65):
        return None
    if bias_side == "bearish" and (bear is None or bear < 65):
        return None

    trap_hits = []
    for tf in ["4h", "1h", "5m", "1m"]:
        trap = state.get(tf, {}).get("trap", {})
        if trap.get("trap"):
            trap_hits.append(f"{tf}: {trap.get('type')} ({trap.get('confidence')})")

    lines = [
        "⚠ BTC Setup detectado",
        "",
        f"Bias: {bias_side} | Bull {bull}% / Bear {bear}%",
        f"Target radar: {target}",
    ]

    if trap_hits:
        lines.append("Trap detector:")
        lines.extend(trap_hits)

    return "\n".join(lines)


def maybe_send_telegram(data: dict):
    if not TELEGRAM_ENABLED:
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram habilitado pero faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return

    msg = build_telegram_message(data)
    if msg:
        print("Telegram READY:")
        print(msg)


# =========================================================
# ZONA 5 · CONSTRUCCIÓN DEL JSON FINAL
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

    for tf in TIMEFRAMES:
        tf_data = fetch(tf)

        if tf_data is None:
            print(f"[SKIP] {tf} sin datos suficientes")
            continue

        data["series"][tf] = tf_data["series"]

        data["state"][tf] = {
            "liquidity": tf_data["liquidity"],
            "sweep": tf_data["sweep"],
            "compression": tf_data["compression"],
            "magnet": tf_data["magnet"],
            "vacuum": tf_data["vacuum"],
            "trap": tf_data["trap"]
        }

    if data["state"]:
        data["bias"] = compute_bias(data["state"])
    else:
        data["bias"] = {
            "bias": "neutral",
            "bullish_pct": 50.0,
            "bearish_pct": 50.0,
            "target": None
        }

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    maybe_send_telegram(data)

    print(f"BTC data updated: {DATA_PATH}")


# =========================================================
# ZONA 6 · ARRANQUE PRINCIPAL
# =========================================================
if __name__ == "__main__":
    main()
