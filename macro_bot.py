import os
import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ======================
# CONFIG (env variables)
# ======================
DEBUG_TEST_SIGNAL = os.getenv("DEBUG_TEST_SIGNAL", "0") == "1"

# Marco principal (señal)
TIMEFRAME = os.getenv("TIMEFRAME", "60m")   # señal por cruce
PERIOD = os.getenv("PERIOD", "30d")

# Marco rápido (confirmación al inicio de la vela siguiente)
EXEC_TIMEFRAME = os.getenv("EXEC_TIMEFRAME", "5m")   # para “ver dirección”
EXEC_PERIOD = os.getenv("EXEC_PERIOD", "7d")

GOLD_TICKER = os.getenv("GOLD_TICKER", "GC=F")
DXY_TICKER = os.getenv("DXY_TICKER", "DX-Y.NYB")
TNX_TICKER = os.getenv("TNX_TICKER", "^TNX")
NASDAQ_TICKER = os.getenv("NASDAQ_TICKER", "^IXIC")

DATA_PATH = os.getenv("MACRO_DATA_PATH", "docs/macro_data.json")
MAX_POINTS = int(os.getenv("MAX_POINTS", "500"))

# Confirmación: % desde el precio del cruce (ej 0.30% = 0.003)
CONFIRM_PCT = float(os.getenv("CONFIRM_PCT", "0.003"))

# Stop “tolerancia” pegada al cruce (ej 0.15% = 0.0015)
SL_PCT = float(os.getenv("SL_PCT", "0.0015"))

# Esperar “un instante” al inicio de la vela nueva (n velas de 5m)
WARMUP_BARS = int(os.getenv("WARMUP_BARS", "1"))  # 2 velas de 5m = 10 min


# ==============
# INDICATORS
# ==============
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.bfill().fillna(50)


# ==============
# DATA IO
# ==============
def fetch_close(ticker: str, period: str, interval: str) -> pd.Series:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Sin datos para {ticker} ({interval}/{period})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).dropna()
    if "close" not in df.columns:
        raise RuntimeError(f"Falta close en {ticker}. Columnas: {list(df.columns)}")
    return df["close"].dropna()


def last_closed_index(series: pd.Series) -> int:
    # para evitar la vela “en formación” cuando hay suficiente histórico
    return -2 if len(series) >= 3 else -1


def load_data(path: str) -> dict:
    if not os.path.exists(path):
        return {"meta": {}, "series": [], "signals": [], "state": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("signals", [])
            data.setdefault("series", [])
            data.setdefault("meta", {})
            data.setdefault("state", {})
            return data
    except Exception:
        return {"meta": {}, "series": [], "signals": [], "state": {}}


def save_data(path: str, data: dict) -> None:
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==============
# SIGNALS
# ==============
def add_signal(signals_list: list, ts_: str, asset: str, type_: str, reason: str,
               strength: int, price: float, extra: dict | None = None) -> None:
    # dedupe por vela: ts + asset + type
    key = (ts_, asset, type_)
    for s in signals_list:
        if (s.get("ts"), s.get("asset"), s.get("type")) == key:
            return

    item = {
        "ts": ts_,
        "asset": asset,
        "type": type_,
        "reason": reason,
        "strength": int(strength),
        "price": round(float(price), 2),
    }
    if extra and isinstance(extra, dict):
        item.update(extra)
    signals_list.append(item)


def pct_move(now: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (now - base) / base


# ==============
# MAIN
# ==============
def main():
    # ---------
    # 1) Marco principal (1H): señal por cruce
    # ---------
    gold = fetch_close(GOLD_TICKER, PERIOD, TIMEFRAME)
    dxy = fetch_close(DXY_TICKER, PERIOD, TIMEFRAME)
    tnx = fetch_close(TNX_TICKER, PERIOD, TIMEFRAME)
    nas = fetch_close(NASDAQ_TICKER, PERIOD, TIMEFRAME)

    i_g = last_closed_index(gold)
    i_d = last_closed_index(dxy)
    i_t = last_closed_index(tnx)
    i_n = last_closed_index(nas)

    # Timestamp REAL de la vela cerrada (clave anti-duplicados)
    candle_ts = pd.Timestamp(gold.index[i_g]).to_pydatetime().replace(tzinfo=timezone.utc).isoformat()

    g = float(gold.iloc[i_g])
    g_prev = float(gold.iloc[i_g - 1]) if len(gold) >= 3 else g

    g_ema21 = float(ema(gold, 21).iloc[i_g])
    ema21_prev = float(ema(gold, 21).iloc[i_g - 1]) if len(gold) >= 3 else g_ema21

    g_ema50 = float(ema(gold, 50).iloc[i_g])

    d = float(dxy.iloc[i_d]);  d_ema21 = float(ema(dxy, 21).iloc[i_d])
    t = float(tnx.iloc[i_t]);  t_ema21 = float(ema(tnx, 21).iloc[i_t])
    n = float(nas.iloc[i_n]);  n_ema21 = float(ema(nas, 21).iloc[i_n])

    gold_rsi = rsi(gold, 14)
    g_rsi14 = float(gold_rsi.iloc[i_g])
    g_rsi14_prev = float(gold_rsi.iloc[i_g - 1]) if len(gold) >= 3 else g_rsi14

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

    now_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    row = {
        "ts": now_ts,
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
            "nasdaq": NASDAQ_TICKER,
        },
    }

    data = load_data(DATA_PATH)
    series = data.get("series", [])
    signals = data.get("signals", [])
    state = data.get("state", {})

    # Limpia señales de prueba antiguas
    signals = [s for s in signals if s.get("reason") != "TEST SIGNAL"]

    # ---------
    # 2) Señal por cruce (1H)
    # ---------
    just_types = set()

    # BUY: cruza de abajo a arriba EMA21
    if (g_prev <= ema21_prev) and (g > g_ema21):
        add_signal(
            signals,
            candle_ts,
            "GOLD",
            "BUY",
            "Cruce alcista: precio pasa por encima de EMA21",
            2,
            g,
            extra={
                "from_candle_ts": candle_ts,
                "ema21": round(g_ema21, 2),
                "stop_price": round(g * (1 - SL_PCT), 2),   # stop sugerido (pegado)
                "confirm_pct": CONFIRM_PCT
            }
        )
        just_types.add("BUY")

    # SELL: cruza de arriba a abajo EMA21
    if (g_prev >= ema21_prev) and (g < g_ema21):
        add_signal(
            signals,
            candle_ts,
            "GOLD",
            "SELL",
            "Cruce bajista: precio cae por debajo de EMA21",
            2,
            g,
            extra={
                "from_candle_ts": candle_ts,
                "ema21": round(g_ema21, 2),
                "stop_price": round(g * (1 + SL_PCT), 2),   # stop sugerido (pegado)
                "confirm_pct": CONFIRM_PCT
            }
        )
        just_types.add("SELL")

    # ---------
    # 3) Preparar confirmación desde el precio del cruce
    #    y “esperar un instante” al inicio de la vela siguiente (5m)
    # ---------
    if "BUY" in just_types:
        state["pending_confirm"] = {
            "direction": "UP",
            "from_ts": candle_ts,
            "from_price": float(g),       # ✅ desde el precio del cruce
            "threshold_pct": CONFIRM_PCT,
            "stop_price": float(g) * (1 - SL_PCT),
            "warmup_bars": WARMUP_BARS,
            "source_type": "BUY",
            "created_at": now_ts
        }

    if "SELL" in just_types:
        state["pending_confirm"] = {
            "direction": "DOWN",
            "from_ts": candle_ts,
            "from_price": float(g),       # ✅ desde el precio del cruce
            "threshold_pct": CONFIRM_PCT,
            "stop_price": float(g) * (1 + SL_PCT),
            "warmup_bars": WARMUP_BARS,
            "source_type": "SELL",
            "created_at": now_ts
        }

    # Si hay confirmación pendiente, miramos marco 5m:
    # - esperamos WARMUP_BARS velas
    # - luego medimos si el precio se mueve en la dirección esperada
    pc = state.get("pending_confirm")
    if pc:
        # data 5m (para “ver dirección” al empezar la vela nueva)
        exec_gold = fetch_close(GOLD_TICKER, EXEC_PERIOD, EXEC_TIMEFRAME)

        # usamos vela cerrada de 5m (también evitar “en formación”)
        i_e = last_closed_index(exec_gold)

        # tomamos “warmup” mirando el cambio desde hace N velas de 5m
        warm = int(pc.get("warmup_bars", WARMUP_BARS))
        if len(exec_gold) >= (abs(i_e) + warm + 1):
            base_idx = i_e - warm
        else:
            base_idx = i_e

        p_now = float(exec_gold.iloc[i_e])
        p_base = float(exec_gold.iloc[base_idx])

        from_price = float(pc.get("from_price", p_base))
        thr = float(pc.get("threshold_pct", CONFIRM_PCT))
        stop_price = float(pc.get("stop_price", from_price))

        move = p_now - from_price
        move_pct = pct_move(p_now, from_price)

        # Si toca stop (protección cerca)
        if pc.get("direction") == "DOWN" and p_now > stop_price:
            add_signal(
                signals,
                now_ts,
                "GOLD",
                "STOP_HIT",
                f"Stop tocado (SELL): precio sube a {p_now:.2f} (> {stop_price:.2f})",
                4,
                p_now,
                extra={"from_price": round(from_price, 2), "stop_price": round(stop_price, 2), "source_ts": pc.get("from_ts")}
            )
            state.pop("pending_confirm", None)

        elif pc.get("direction") == "UP" and p_now < stop_price:
            add_signal(
                signals,
                now_ts,
                "GOLD",
                "STOP_HIT",
                f"Stop tocado (BUY): precio baja a {p_now:.2f} (< {stop_price:.2f})",
                4,
                p_now,
                extra={"from_price": round(from_price, 2), "stop_price": round(stop_price, 2), "source_ts": pc.get("from_ts")}
            )
            state.pop("pending_confirm", None)

        else:
            # Confirmación por % (desde el cruce)
            if pc.get("direction") == "DOWN" and move_pct <= -thr:
                add_signal(
                    signals,
                    now_ts,
                    "GOLD",
                    "CONFIRM_DOWN",
                    f"Confirmación bajista: {move_pct*100:.2f}% ({move:.2f}$) desde cruce {from_price:.2f}",
                    3,
                    p_now,
                    extra={
                        "from_price": round(from_price, 2),
                        "move": round(move, 2),
                        "move_pct": round(move_pct, 6),
                        "threshold_pct": thr,
                        "stop_price": round(stop_price, 2),
                        "source_ts": pc.get("from_ts"),
                        "source_type": pc.get("source_type"),
                        "exec_timeframe": EXEC_TIMEFRAME,
                        "warmup_bars": warm,
                        "p_base": round(p_base, 2),
                    },
                )
                state.pop("pending_confirm", None)

            elif pc.get("direction") == "UP" and move_pct >= thr:
                add_signal(
                    signals,
                    now_ts,
                    "GOLD",
                    "CONFIRM_UP",
                    f"Confirmación alcista: {move_pct*100:.2f}% (+{move:.2f}$) desde cruce {from_price:.2f}",
                    3,
                    p_now,
                    extra={
                        "from_price": round(from_price, 2),
                        "move": round(move, 2),
                        "move_pct": round(move_pct, 6),
                        "threshold_pct": thr,
                        "stop_price": round(stop_price, 2),
                        "source_ts": pc.get("from_ts"),
                        "source_type": pc.get("source_type"),
                        "exec_timeframe": EXEC_TIMEFRAME,
                        "warmup_bars": warm,
                        "p_base": round(p_base, 2),
                    },
                )
                state.pop("pending_confirm", None)

    # TEST opcional
    if DEBUG_TEST_SIGNAL:
        add_signal(signals, now_ts, "GOLD", "WARN", "TEST SIGNAL", 1, g)

    # ---------
    # 4) Guardar datos
    # ---------
    series.append(row)
    if len(series) > MAX_POINTS:
        series = series[-MAX_POINTS:]

    data["meta"] = {"timeframe": TIMEFRAME, "period": PERIOD, "updated_utc": now_ts}
    data["series"] = series
    data["signals"] = signals
    data["state"] = state

    save_data(DATA_PATH, data)
    print(f"Saved dashboard data: {DATA_PATH} (points={len(series)}) | signals={len(signals)}")


if __name__ == "__main__":
    main()
