import os
import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ======================
# CONFIG (env variables)
# ======================
DEBUG_TEST_SIGNAL = os.getenv("DEBUG_TEST_SIGNAL", "0") == "1"

# Marco principal (estructura)
TIMEFRAME = os.getenv("TIMEFRAME", "60m")
PERIOD = os.getenv("PERIOD", "30d")

# Marco rápido (confirmación táctica)
EXEC_TIMEFRAME = os.getenv("EXEC_TIMEFRAME", "5m")
EXEC_PERIOD = os.getenv("EXEC_PERIOD", "7d")

# Tickers macro
GOLD_TICKER = os.getenv("GOLD_TICKER", "GC=F")
DXY_TICKER = os.getenv("DXY_TICKER", "DX-Y.NYB")
TNX_TICKER = os.getenv("TNX_TICKER", "^TNX")
NASDAQ_TICKER = os.getenv("NASDAQ_TICKER", "^IXIC")

# Output macro dashboard (antiguo)
DATA_PATH = os.getenv("MACRO_DATA_PATH", "docs/macro_data.json")
MAX_POINTS = int(os.getenv("MAX_POINTS", "500"))

# Confirmación / protección
CONFIRM_PCT = float(os.getenv("CONFIRM_PCT", "0.003"))   # 0.30%
SL_PCT = float(os.getenv("SL_PCT", "0.0015"))            # 0.15% stop pegado
WARMUP_BARS = int(os.getenv("WARMUP_BARS", "1"))         # 1 vela de 5m = 5 min

# ==== COMPRESIÓN / EXPANSIÓN (PRO) ====
COMP_EMA_PCT = float(os.getenv("COMP_EMA_PCT", "0.0015"))  # 0.15% EMA21-EMA50 pegadas
COMP_ATR_PCT = float(os.getenv("COMP_ATR_PCT", "0.0018"))  # 0.18% ATR% bajo
SLOPE_PCT = float(os.getenv("SLOPE_PCT", "0.0002"))        # 0.02% EMA21 casi plana
ATR_LEN = int(os.getenv("ATR_LEN", "14"))


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


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ==============
# DATA FETCH
# ==============
def fetch_ohlc(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Sin datos para {ticker} ({interval}/{period})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).dropna()

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Faltan OHLC en {ticker}. Columnas: {list(df.columns)}")

    return df[["open", "high", "low", "close"]].dropna()


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
    # Evita la vela en formación cuando hay histórico suficiente
    return -2 if len(series) >= 3 else -1


# ==============
# JSON IO
# ==============
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
               strength: int, price: float, extra: dict = None) -> None:
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
    if not base:
        return 0.0
    return (now - base) / base


# ==========================
# PREMIUM TERMINAL HELPERS
# ==========================
def load_watchlist_symbols(path: str) -> list[str]:
    """
    Lee docs/premium-terminal/watchlist.json y devuelve lista plana de símbolos.
    Formato esperado:
    {
      "Stocks": [{"symbol":"AAPL"}, ...],
      "Crypto": [{"symbol":"BTC-USD"}, ...],
      ...
    }
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            w = json.load(f)
        out = []
        if isinstance(w, dict):
            for _, items in w.items():
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            sym = it.get("symbol")
                            if sym and isinstance(sym, str):
                                out.append(sym.strip())
        # elimina duplicados manteniendo orden
        seen = set()
        unique = []
        for s in out:
            if s not in seen:
                unique.append(s)
                seen.add(s)
        return unique
    except Exception:
        return []


def classify_volatility(atr_pct_val: float) -> str:
    # simple y robusto
    if atr_pct_val is None:
        return "medium"
    if atr_pct_val <= 0.0018:
        return "low"
    if atr_pct_val >= 0.0030:
        return "high"
    return "medium"


def build_asset_metrics(symbol: str, period: str, interval: str) -> dict | None:
    """
    Métricas básicas para cualquier símbolo de watchlist.
    No rompe si un símbolo falla, devuelve None.
    """
    try:
        df = fetch_ohlc(symbol, period, interval)
        close = df["close"]
        i = last_closed_index(close)
        price = float(close.iloc[i])

        e9  = float(ema(close, 9).iloc[i])
        e21 = float(ema(close, 21).iloc[i])
        # ema50 opcional si hay datos suficientes
        e50_series = ema(close, 50)
        e50 = float(e50_series.iloc[i]) if len(e50_series) else e21

        a = float(atr(df, ATR_LEN).iloc[i])
        atr_pct_val = (a / price) if price else 0.0

        # score simple (para no inventar macro)
        score = 65 if price > e21 else 35
        trend = "bullish" if price > e21 else "bearish"

        # señal simple
        if price > e21 and e21 > e50:
            signal = "explosion_up"
        elif price < e21 and e21 < e50:
            signal = "explosion_down"
        else:
            signal = "neutral"

        return {
            "score": int(score),
            "trend": trend,
            "volatility": classify_volatility(atr_pct_val),
            "signal": signal,
            "price": round(price, 4) if price < 100 else round(price, 2),
            "ema9":  round(e9, 4) if e9 < 100 else round(e9, 2),
            "ema21": round(e21, 4) if e21 < 100 else round(e21, 2),
            "ema50": round(e50, 4) if e50 < 100 else round(e50, 2),
            "atr_pct": round(atr_pct_val, 6),
        }
    except Exception:
        return None


# ==============
# MAIN
# ==============
def main():
    # ========== 1) Datos 1H (estructura) ==========
    gold_df = fetch_ohlc(GOLD_TICKER, PERIOD, TIMEFRAME)
    gold = gold_df["close"]

    dxy = fetch_close(DXY_TICKER, PERIOD, TIMEFRAME)
    tnx = fetch_close(TNX_TICKER, PERIOD, TIMEFRAME)
    nas = fetch_close(NASDAQ_TICKER, PERIOD, TIMEFRAME)

    i_g = last_closed_index(gold)
    i_d = last_closed_index(dxy)
    i_t = last_closed_index(tnx)
    i_n = last_closed_index(nas)

    candle_ts = pd.Timestamp(gold.index[i_g]).to_pydatetime().replace(tzinfo=timezone.utc).isoformat()
    now_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    g = float(gold.iloc[i_g])
    g_prev = float(gold.iloc[i_g - 1]) if len(gold) >= 3 else g

    ema9_series  = ema(gold, 9)
    ema21_series = ema(gold, 21)
    ema50_series = ema(gold, 50)

    g_ema9  = float(ema9_series.iloc[i_g])
    g_ema21 = float(ema21_series.iloc[i_g])
    g_ema50 = float(ema50_series.iloc[i_g])

    ema21_prev = float(ema21_series.iloc[i_g - 1]) if len(gold) >= 3 else g_ema21
    ema21_slope_pct = ((g_ema21 - ema21_prev) / g) if g else 0.0

    gold_rsi = rsi(gold, 14)
    g_rsi14 = float(gold_rsi.iloc[i_g])
    g_rsi14_prev = float(gold_rsi.iloc[i_g - 1]) if len(gold) >= 3 else g_rsi14

    d = float(dxy.iloc[i_d]);  d_ema21 = float(ema(dxy, 21).iloc[i_d])
    t = float(tnx.iloc[i_t]);  t_ema21 = float(ema(tnx, 21).iloc[i_t])
    n = float(nas.iloc[i_n]);  n_ema21 = float(ema(nas, 21).iloc[i_n])

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

    gold_atr = atr(gold_df, ATR_LEN)
    g_atr = float(gold_atr.iloc[i_g])
    atr_pct = (g_atr / g) if g else 0.0

    ema_spread = abs(g_ema21 - g_ema50)
    ema_spread_pct = (ema_spread / g) if g else 0.0

    # ========== 2) Cargar JSON macro dashboard ==========
    data = load_data(DATA_PATH)
    series = data.get("series", [])
    signals = data.get("signals", [])
    state = data.get("state", {})

    signals = [s for s in signals if s.get("reason") != "TEST SIGNAL"]

    # ========== 3) Guardar punto de serie ==========
    row = {
        "ts": now_ts,
        "candle_ts": candle_ts,
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
        "ema_spread_pct": round(ema_spread_pct, 6),
        "atr": round(g_atr, 4),
        "atr_pct": round(atr_pct, 6),
        "ema21_slope_pct": round(ema21_slope_pct, 6),
        "tickers": {"gold": GOLD_TICKER, "dxy": DXY_TICKER, "tnx": TNX_TICKER, "nasdaq": NASDAQ_TICKER},
    }

    # ========== 4) Señales por CRUCE (1H) ==========
    just_types = set()

    if (g_prev <= ema21_prev) and (g > g_ema21):
        add_signal(
            signals, candle_ts, "GOLD", "BUY",
            "Cruce alcista: precio pasa por encima de EMA21",
            2, g,
            extra={"ema21": round(g_ema21, 2), "stop_price": round(g * (1 - SL_PCT), 2), "confirm_pct": CONFIRM_PCT},
        )
        just_types.add("BUY")

    if (g_prev >= ema21_prev) and (g < g_ema21):
        add_signal(
            signals, candle_ts, "GOLD", "SELL",
            "Cruce bajista: precio cae por debajo de EMA21",
            2, g,
            extra={"ema21": round(g_ema21, 2), "stop_price": round(g * (1 + SL_PCT), 2), "confirm_pct": CONFIRM_PCT},
        )
        just_types.add("SELL")

    # ========== 5) PRO: COMPRESIÓN / EXPANSIÓN ==========
    is_ema_tight = ema_spread_pct <= COMP_EMA_PCT
    is_atr_low = atr_pct <= COMP_ATR_PCT
    is_flat = abs(ema21_slope_pct) <= SLOPE_PCT
    compression_now = bool(is_ema_tight and is_atr_low and is_flat)

    prev_comp = bool(state.get("compression_active", False))

    state["metrics"] = {
        "ema_spread": round(ema_spread, 4),
        "ema_spread_pct": round(ema_spread_pct, 6),
        "atr": round(g_atr, 4),
        "atr_pct": round(atr_pct, 6),
        "ema21_slope_pct": round(ema21_slope_pct, 6),
        "thresholds": {"COMP_EMA_PCT": COMP_EMA_PCT, "COMP_ATR_PCT": COMP_ATR_PCT, "SLOPE_PCT": SLOPE_PCT, "ATR_LEN": ATR_LEN},
    }

    if compression_now and not prev_comp:
        state["compression_active"] = True
        state["compression_from_ts"] = candle_ts
        state["compression_from_price"] = float(g)
        add_signal(
            signals, candle_ts, "GOLD", "COMPRESSION_ON",
            "EMAs hermanadas + ATR bajo + EMA21 plana (energía cargándose)",
            1, g,
            extra={"ema_spread_pct": round(ema_spread_pct, 6), "atr_pct": round(atr_pct, 6)},
        )

    if (not compression_now) and prev_comp:
        state["compression_active"] = False
        add_signal(
            signals, candle_ts, "GOLD", "COMPRESSION_OFF",
            "Fin de compresión (empieza expansión)",
            1, g,
            extra={"ema_spread_pct": round(ema_spread_pct, 6), "atr_pct": round(atr_pct, 6)},
        )

        if (g > g_ema21) and (ema21_slope_pct > SLOPE_PCT):
            state["market_state"] = "EXPANSION_UP"
            add_signal(signals, candle_ts, "GOLD", "EXPANSION_UP",
                       "Inicio expansión alcista tras compresión (posible movimiento fuerte)",
                       2, g, extra={"ema21_slope_pct": round(ema21_slope_pct, 6)})
        elif (g < g_ema21) and (ema21_slope_pct < -SLOPE_PCT):
            state["market_state"] = "EXPANSION_DOWN"
            add_signal(signals, candle_ts, "GOLD", "EXPANSION_DOWN",
                       "Inicio expansión bajista tras compresión (posible movimiento fuerte)",
                       2, g, extra={"ema21_slope_pct": round(ema21_slope_pct, 6)})
        else:
            state["market_state"] = "EXPANSION_UNCLEAR"

    state["market_state"] = "COMPRESSION" if compression_now else state.get("market_state", "NORMAL")

    # ========== 6) Confirmación táctica (5m) desde el cruce ==========
    if "BUY" in just_types:
        state["pending_confirm"] = {
            "direction": "UP",
            "from_ts": candle_ts,
            "from_price": float(g),
            "threshold_pct": CONFIRM_PCT,
            "stop_price": float(g) * (1 - SL_PCT),
            "warmup_bars": WARMUP_BARS,
            "source_type": "BUY",
            "created_at": now_ts,
        }

    if "SELL" in just_types:
        state["pending_confirm"] = {
            "direction": "DOWN",
            "from_ts": candle_ts,
            "from_price": float(g),
            "threshold_pct": CONFIRM_PCT,
            "stop_price": float(g) * (1 + SL_PCT),
            "warmup_bars": WARMUP_BARS,
            "source_type": "SELL",
            "created_at": now_ts,
        }

    pc = state.get("pending_confirm")
    if pc:
        exec_gold = fetch_close(GOLD_TICKER, EXEC_PERIOD, EXEC_TIMEFRAME)
        i_e = last_closed_index(exec_gold)

        warm = int(pc.get("warmup_bars", WARMUP_BARS))
        if len(exec_gold) >= (warm + 2):
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

        if pc.get("direction") == "DOWN" and p_now > stop_price:
            add_signal(signals, now_ts, "GOLD", "STOP_HIT",
                       f"Stop tocado (SELL): {p_now:.2f} > {stop_price:.2f}",
                       4, p_now,
                       extra={"from_price": round(from_price, 2), "stop_price": round(stop_price, 2), "source_ts": pc.get("from_ts")})
            state.pop("pending_confirm", None)

        elif pc.get("direction") == "UP" and p_now < stop_price:
            add_signal(signals, now_ts, "GOLD", "STOP_HIT",
                       f"Stop tocado (BUY): {p_now:.2f} < {stop_price:.2f}",
                       4, p_now,
                       extra={"from_price": round(from_price, 2), "stop_price": round(stop_price, 2), "source_ts": pc.get("from_ts")})
            state.pop("pending_confirm", None)

        else:
            if pc.get("direction") == "DOWN" and move_pct <= -thr:
                add_signal(signals, now_ts, "GOLD", "CONFIRM_DOWN",
                           f"Confirmación bajista: {move_pct*100:.2f}% ({move:.2f}$) desde cruce {from_price:.2f}",
                           3, p_now,
                           extra={"from_price": round(from_price, 2), "move": round(move, 2), "move_pct": round(move_pct, 6),
                                  "threshold_pct": thr, "stop_price": round(stop_price, 2),
                                  "source_ts": pc.get("from_ts"), "source_type": pc.get("source_type"),
                                  "exec_timeframe": EXEC_TIMEFRAME, "warmup_bars": warm, "p_base": round(p_base, 2)})
                state.pop("pending_confirm", None)

            elif pc.get("direction") == "UP" and move_pct >= thr:
                add_signal(signals, now_ts, "GOLD", "CONFIRM_UP",
                           f"Confirmación alcista: {move_pct*100:.2f}% (+{move:.2f}$) desde cruce {from_price:.2f}",
                           3, p_now,
                           extra={"from_price": round(from_price, 2), "move": round(move, 2), "move_pct": round(move_pct, 6),
                                  "threshold_pct": thr, "stop_price": round(stop_price, 2),
                                  "source_ts": pc.get("from_ts"), "source_type": pc.get("source_type"),
                                  "exec_timeframe": EXEC_TIMEFRAME, "warmup_bars": warm, "p_base": round(p_base, 2)})
                state.pop("pending_confirm", None)

    if DEBUG_TEST_SIGNAL:
        add_signal(signals, now_ts, "GOLD", "WARN", "TEST SIGNAL", 1, g)

    # ========== 7) Guardar macro dashboard ==========
    series.append(row)
    if len(series) > MAX_POINTS:
        series = series[-MAX_POINTS:]

    data["meta"] = {"timeframe": TIMEFRAME, "period": PERIOD, "updated_utc": now_ts}
    data["series"] = series
    data["signals"] = signals
    data["state"] = state

    save_data(DATA_PATH, data)
    print(f"Saved dashboard data: {DATA_PATH} (points={len(series)}) | signals={len(signals)}")

    # ==========================
    # EXPORT para Premium Terminal (assets)
    # ==========================
    premium_path = "docs/premium-terminal/macro_data.json"

    # construimos assets SOLO con lo que ya calculas aquí (macro base)
    premium_assets = {
        GOLD_TICKER: {
            "score": score,
            "trend": "bullish" if g_ema21 > g_ema50 else "bearish",
            "volatility": "low" if atr_pct <= COMP_ATR_PCT else ("high" if atr_pct >= 0.003 else "medium"),
            "signal": "explosion_up" if (g > g_ema21 and g_ema21 > g_ema50) else ("explosion_down" if (g < g_ema21 and g_ema21 < g_ema50) else "neutral"),
            "price": round(g, 2),
            "ema9": round(g_ema9, 2),
            "ema21": round(g_ema21, 2),
            "ema50": round(g_ema50, 2),
            "atr_pct": round(atr_pct, 6),
        },
        DXY_TICKER: {
            "score": 50,
            "trend": "bullish" if d > d_ema21 else "bearish",
            "volatility": "medium",
            "signal": "neutral",
            "price": round(d, 4),
            "ema21": round(d_ema21, 4),
        },
        TNX_TICKER: {
            "score": 50,
            "trend": "bullish" if t > t_ema21 else "bearish",
            "volatility": "medium",
            "signal": "neutral",
            "price": round(t, 4),
            "ema21": round(t_ema21, 4),
        },
        NASDAQ_TICKER: {
            "score": 50,
            "trend": "bullish" if n > n_ema21 else "bearish",
            "volatility": "medium",
            "signal": "neutral",
            "price": round(n, 2),
            "ema21": round(n_ema21, 2),
        },
    }

    # ========= Añadir watchlist (BTC, acciones, etc.) =========
    watchlist_path = "docs/premium-terminal/watchlist.json"
    symbols = load_watchlist_symbols(watchlist_path)

    for sym in symbols:
        # si ya existe (por ejemplo GC=F), no lo machacamos
        if sym in premium_assets:
            continue

        m = build_asset_metrics(sym, PERIOD, TIMEFRAME)
        if m:
            premium_assets[sym] = m

    # === mini histórico para pintar EMAs (últimos 120 puntos) ===
tail = series[-120:] if len(series) > 120 else series

premium_series = {
    GOLD_TICKER: [
        {
            "ts": r.get("ts"),
            "price": r.get("gold"),
            "ema21": r.get("gold_ema21"),
            "ema50": r.get("gold_ema50"),
            # ema9 no existe en row todavía -> (por ahora no lo metemos aquí)
        }
        for r in tail
        if r.get("gold") is not None
    ]
}

    premium_out = {
        "meta": {"updated_utc": now_ts, "timeframe": "1h", "source": "macro_bot.py"},
        "assets": premium_assets,
        "series": premium_series,
    }

    os.makedirs(os.path.dirname(premium_path), exist_ok=True)
    with open(premium_path, "w", encoding="utf-8") as f:
        json.dump(premium_out, f, ensure_ascii=False, indent=2)

    print(f"Saved premium terminal data: {premium_path} | assets={len(premium_assets)}")


if __name__ == "__main__":
    main()
