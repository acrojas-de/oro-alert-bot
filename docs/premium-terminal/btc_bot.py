import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
import numpy as np


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


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    return out.fillna(50)


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
# ZONA 2H · ALERT ENGINE POR TIMEFRAME
# =========================================================
def detect_divergence(close: pd.Series, rsi_series: pd.Series, lookback: int = 8) -> dict:
    s_close = close.dropna()
    s_rsi = rsi_series.dropna()

    size = min(len(s_close), len(s_rsi))
    if size < lookback * 3:
        return {
            "active": False,
            "type": "none",
            "label": "insufficient_data"
        }

    s_close = s_close.iloc[-lookback * 3:]
    s_rsi = s_rsi.iloc[-lookback * 3:]

    a_close = s_close.iloc[:lookback]
    b_close = s_close.iloc[lookback:lookback * 2]
    c_close = s_close.iloc[lookback * 2:]

    a_rsi = s_rsi.iloc[:lookback]
    b_rsi = s_rsi.iloc[lookback:lookback * 2]
    c_rsi = s_rsi.iloc[lookback * 2:]

    bearish = (
        float(c_close.max()) > float(b_close.max()) > float(a_close.max())
        and float(c_rsi.max()) < float(b_rsi.max())
    )

    bullish = (
        float(c_close.min()) < float(b_close.min()) < float(a_close.min())
        and float(c_rsi.min()) > float(b_rsi.min())
    )

    if bearish:
        return {
            "active": True,
            "type": "bearish",
            "label": "bearish_divergence"
        }

    if bullish:
        return {
            "active": True,
            "type": "bullish",
            "label": "bullish_divergence"
        }

    return {
        "active": False,
        "type": "none",
        "label": "no_divergence"
    }


def detect_directional_bias_tf(
    close: pd.Series,
    ema21: pd.Series,
    ema50: pd.Series,
    rsi_series: pd.Series,
    liq: dict,
    sweep: dict,
    compression: dict,
    magnet: dict,
    divergence: dict
) -> dict:
    if len(close) == 0 or len(ema21) == 0 or len(ema50) == 0 or len(rsi_series) == 0:
        return {
            "score": 0,
            "bias": "neutral",
            "confidence": "low",
            "reasons": ["insufficient_data"]
        }

    last_close = float(close.iloc[-1])
    last_ema21 = float(ema21.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_rsi = float(rsi_series.iloc[-1])

    score = 0
    reasons = []

    if last_ema21 > last_ema50:
        score += 2
        reasons.append("ema21_above_ema50")
    elif last_ema21 < last_ema50:
        score -= 2
        reasons.append("ema21_below_ema50")

    if last_close > last_ema21:
        score += 1
        reasons.append("price_above_ema21")
    elif last_close < last_ema21:
        score -= 1
        reasons.append("price_below_ema21")

    if last_rsi >= 55:
        score += 1
        reasons.append("rsi_bullish")
    elif last_rsi <= 45:
        score -= 1
        reasons.append("rsi_bearish")

    if magnet.get("direction") == "up":
        score += 1
        reasons.append("magnet_up")
    elif magnet.get("direction") == "down":
        score -= 1
        reasons.append("magnet_down")

    if sweep.get("sweep_low"):
        score += 1
        reasons.append("sweep_low_supports_up")

    if sweep.get("sweep_high"):
        score -= 1
        reasons.append("sweep_high_supports_down")

    if divergence.get("type") == "bullish":
        score += 1
        reasons.append("bullish_divergence_bonus")
    elif divergence.get("type") == "bearish":
        score -= 1
        reasons.append("bearish_divergence_penalty")

    compression_active = bool(compression.get("compression", False))

    if score >= 3:
        bias = "bullish"
        confidence = "high" if compression_active else "medium"
    elif score <= -3:
        bias = "bearish"
        confidence = "high" if compression_active else "medium"
    else:
        bias = "neutral"
        confidence = "low"

    return {
        "score": int(score),
        "bias": bias,
        "confidence": confidence,
        "reasons": reasons
    }


def confirm_breakout(df: pd.DataFrame, compression: dict, lookback: int = 12, min_body_ratio: float = 0.55) -> dict:
    if df is None or df.empty or len(df) < lookback + 2:
        return {
            "active": False,
            "direction": "none",
            "label": "insufficient_data"
        }

    if not bool(compression.get("compression", False)):
        return {
            "active": False,
            "direction": "none",
            "label": "no_compression_context"
        }

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return {
            "active": False,
            "direction": "none",
            "label": "missing_ohlc"
        }

    box = df.iloc[-(lookback + 1):-1]
    last = df.iloc[-1]

    box_high = float(box["high"].max())
    box_low = float(box["low"].min())

    open_ = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])
    close_ = float(last["close"])

    candle_range = max(high - low, 1e-9)
    body = abs(close_ - open_)
    body_ratio = body / candle_range

    bull_breakout = close_ > box_high and body_ratio >= min_body_ratio
    bear_breakout = close_ < box_low and body_ratio >= min_body_ratio

    if bull_breakout:
        return {
            "active": True,
            "direction": "up",
            "body_ratio": round(body_ratio, 4),
            "box_high": round(box_high, 6),
            "box_low": round(box_low, 6),
            "label": "bullish_breakout_confirmed"
        }

    if bear_breakout:
        return {
            "active": True,
            "direction": "down",
            "body_ratio": round(body_ratio, 4),
            "box_high": round(box_high, 6),
            "box_low": round(box_low, 6),
            "label": "bearish_breakout_confirmed"
        }

    return {
        "active": False,
        "direction": "none",
        "body_ratio": round(body_ratio, 4),
        "box_high": round(box_high, 6),
        "box_low": round(box_low, 6),
        "label": "no_breakout"
    }


def build_alert_engine_tf(
    df: pd.DataFrame,
    close: pd.Series,
    ema21: pd.Series,
    ema50: pd.Series,
    rsi_series: pd.Series,
    liq: dict,
    sweep: dict,
    compression: dict,
    magnet: dict
) -> dict:
    divergence = detect_divergence(close, rsi_series)

    directional_bias = detect_directional_bias_tf(
        close=close,
        ema21=ema21,
        ema50=ema50,
        rsi_series=rsi_series,
        liq=liq,
        sweep=sweep,
        compression=compression,
        magnet=magnet,
        divergence=divergence
    )

    breakout = confirm_breakout(df, compression)

    return {
        "compression": {
            "active": bool(compression.get("compression", False)),
            "volatility": round(float(compression.get("volatility", 0.0)), 8),
            "macd_compression": round(float(compression.get("macd_compression", 0.0)), 6),
            "label": "compression" if compression.get("compression") else "normal"
        },
        "directional_bias": directional_bias,
        "divergence": divergence,
        "breakout_confirmation": breakout
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
    rsi14 = rsi(close, 14)
    macd_line, signal, hist = macd(close)

    liq = liquidity_levels(close)
    sweep = detect_liquidity_sweep(close, liq)
    compression = detect_compression(close, hist)
    magnet = detect_liquidity_magnet(close, liq)
    vacuum = detect_liquidity_vacuum(close, liq)
    trap = detect_trap(sweep, magnet, compression)

    alert_engine = build_alert_engine_tf(
        df=df,
        close=close,
        ema21=ema21,
        ema50=ema50,
        rsi_series=rsi14,
        liq=liq,
        sweep=sweep,
        compression=compression,
        magnet=magnet
    )

    out = []
    for i in range(len(close)):
        if pd.isna(close.iloc[i]) or pd.isna(ema21.iloc[i]) or pd.isna(ema50.iloc[i]):
            continue

        out.append({
            "price": round(float(close.iloc[i]), 6),
            "ema21": round(float(ema21.iloc[i]), 6),
            "ema50": round(float(ema50.iloc[i]), 6),
            "rsi": round(float(rsi14.iloc[i]), 6) if pd.notna(rsi14.iloc[i]) else None,
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
        "trap": trap,
        "alert_engine": alert_engine
    }


# =========================================================
# ZONA 4 · TELEGRAM INTELIGENTE (PREPARADO)
# =========================================================
def build_telegram_message(data: dict) -> str | None:
    bias = data.get("bias", {})
    state = data.get("state", {})

    if not bias or not state:
        return None

    bias_side = bias.get("bias", "neutral")
    bull = bias.get("bullish_pct")
    bear = bias.get("bearish_pct")
    target = bias.get("target")

    lines = [
        "⚠ BTC Alert Engine",
        "",
        f"Bias global: {bias_side} | Bull {bull}% / Bear {bear}%",
        f"Target radar: {target}",
        ""
    ]

    interesting = False

    for tf in ["4h", "1h", "5m", "1m"]:
        tf_state = state.get(tf, {})
        ae = tf_state.get("alert_engine", {})

        compression = ae.get("compression", {})
        directional = ae.get("directional_bias", {})
        divergence = ae.get("divergence", {})
        breakout = ae.get("breakout_confirmation", {})

        tf_lines = []

        if compression.get("active"):
            tf_lines.append(
                f"• {tf} compression | bias {directional.get('bias')} ({directional.get('confidence')})"
            )

        if divergence.get("active"):
            tf_lines.append(f"• {tf} divergence: {divergence.get('type')}")

        if breakout.get("active"):
            tf_lines.append(f"• {tf} breakout confirmed: {breakout.get('direction')}")

        if tf_lines:
            interesting = True
            lines.extend(tf_lines)

    if not interesting:
        return None

    return "\n".join(lines)


def maybe_send_telegram(data: dict):

    if not TELEGRAM_ENABLED:
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram habilitado pero faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
        return

    msg = build_telegram_message(data)

    if not msg:
        return

    try:

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg
        }

        r = requests.post(url, data=payload, timeout=10)

        if r.status_code == 200:
            print("Telegram enviado correctamente")
        else:
            print("Error Telegram:", r.text)

    except Exception as e:
        print("Error enviando Telegram:", e)

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
            "trap": tf_data["trap"],
            "alert_engine": tf_data["alert_engine"]
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

        data["setup"] = compute_trade_setup(data["state"], data["bias"])
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    maybe_send_telegram(data)

    print(f"BTC data updated: {DATA_PATH}")


# =========================================================
# ZONA 6 · ARRANQUE PRINCIPAL
# =========================================================
if __name__ == "__main__":
    main()
