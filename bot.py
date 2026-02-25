import os
import json
import pandas as pd
import requests
import yfinance as yf

DEBUG = os.getenv("DEBUG", "0") == "1"

# ===== CONFIG =====
TIMEFRAME = "60m"
PERIOD = "60d"

GOLD_TICKER = os.getenv("GOLD_TICKER", "GC=F")  # recomendado: GC=F
DXY_TICKER = "DX-Y.NYB"
TNX_TICKER = "^TNX"
NASDAQ_TICKER = "^IXIC"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

STATE_FILE = "state.json"
MAX_DIST_TO_EMA21 = float(os.getenv("MAX_DIST_TO_EMA21", "0.006"))  # 0.6%

# ===== INDICATORS =====
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def crossed_up(a: pd.Series, b: pd.Series, i: int) -> bool:
    return a.iloc[i-1] <= b.iloc[i-1] and a.iloc[i] > b.iloc[i]

def crossed_down(a: pd.Series, b: pd.Series, i: int) -> bool:
    return a.iloc[i-1] >= b.iloc[i-1] and a.iloc[i] < b.iloc[i]

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ===== TELEGRAM =====
def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=20)
    print("Telegram status:", r.status_code)

# ===== DATA =====
def fetch_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=TIMEFRAME, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Sin datos para {ticker}")

    # Si viene MultiIndex (a veces pasa con yfinance), aplana columnas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normaliza nombres de columnas
    df = df.rename(columns=str.lower).dropna()

    # Asegura que existen las columnas necesarias (ATR usa high/low/close)
    needed = ["open", "high", "low", "close"]
    for col in needed:
        if col not in df.columns:
            raise RuntimeError(f"Falta columna '{col}' en datos de {ticker}. Columnas: {list(df.columns)}")

    df = df.dropna()
    return df

def last_closed_index(df: pd.DataFrame) -> int:
    return -2 if len(df) >= 3 else -1

# ===== STATE =====
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)

# ===== MAIN =====
def main():
    if DEBUG:
        send_telegram("🚀 DEBUG: ORO BOT arrancó correctamente")
    state = load_state()
    position = state.get("position", "NONE")  # NONE / LONG / SHORT
    entry = state.get("entry")
    sl = state.get("sl")
    tp = state.get("tp")
    last_sent = state.get("last_sent", "")

    gold = fetch_ohlc(GOLD_TICKER)
    dxy = fetch_ohlc(DXY_TICKER)
    tnx = fetch_ohlc(TNX_TICKER)
    nasdaq = fetch_ohlc(NASDAQ_TICKER)

    i = last_closed_index(gold)

    close = gold["close"]
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    e50 = ema(close, 50)
    _, _, hist = macd(close)
    a = atr(gold, 14)

    price = float(close.iloc[i])
    ema21_v = float(e21.iloc[i])
    ema50_v = float(e50.iloc[i])
    hist_v = float(hist.iloc[i])
    atr_v = float(a.iloc[i]) if pd.notna(a.iloc[i]) else None

    # Contexto DXY
    dxy_i = last_closed_index(dxy)
    dxy_close = dxy["close"]
    dxy_ema21 = ema(dxy_close, 21)
    dxy_weak = dxy_close.iloc[dxy_i] < dxy_ema21.iloc[dxy_i]
    dxy_strong = dxy_close.iloc[dxy_i] > dxy_ema21.iloc[dxy_i]

    # Anti-persecución
    dist = abs(price - ema21_v) / ema21_v

    # Tendencia
    ema50_up = e50.iloc[i] > e50.iloc[i-5]
    ema50_down = e50.iloc[i] < e50.iloc[i-5]
    uptrend = (e21.iloc[i] > e50.iloc[i]) and ema50_up
    downtrend = (e21.iloc[i] < e50.iloc[i]) and ema50_down

    # ===== SALIDAS =====
    if position in ("LONG", "SHORT"):
        high_i = float(gold["high"].iloc[i])
        low_i = float(gold["low"].iloc[i])

        exit_reason = None
        if position == "LONG":
            if tp is not None and high_i >= tp:
                exit_reason = f"✅ TP alcanzado ({tp})"
            elif sl is not None and low_i <= sl:
                exit_reason = f"🛑 SL tocado ({sl})"
            elif crossed_down(e9, e21, i) and hist_v < 0:
                exit_reason = "🔁 Giro bajista: cruce EMA9↓EMA21 + MACD<0"

        if position == "SHORT":
            if tp is not None and low_i <= tp:
                exit_reason = f"✅ TP alcanzado ({tp})"
            elif sl is not None and high_i >= sl:
                exit_reason = f"🛑 SL tocado ({sl})"
            elif crossed_up(e9, e21, i) and hist_v > 0:
                exit_reason = "🔁 Giro alcista: cruce EMA9↑EMA21 + MACD>0"

        if exit_reason:
            msg = (
                f"🟡 CERRAR {position} ORO ({GOLD_TICKER}) 1H\n"
                f"Precio (cierre vela): {price}\n"
                f"Motivo: {exit_reason}\n"
                f"Entrada: {entry}\n"
                f"SL: {sl} | TP: {tp}\n"
            )
            send_telegram(msg)
            state = {"position": "NONE", "entry": None, "sl": None, "tp": None, "last_sent": ""}
            save_state(state)
            return

    # ===== ENTRADAS =====
    if position == "NONE":
        if atr_v is None or atr_v == 0:
            print("ATR no disponible aún.")
            return

        if dist > MAX_DIST_TO_EMA21:
            print("Precio extendido vs EMA21: sin entrada.")
            return

        buy = (
            uptrend
            and crossed_up(e9, e21, i)
            and price > ema21_v
            and hist_v > 0
            and dxy_weak
        )

        sell = (
            downtrend
            and crossed_down(e9, e21, i)
            and price < ema21_v
            and hist_v < 0
            and dxy_strong
        )

        if buy or sell:
            if buy:
                pos_new = "LONG"
                sl_new = price - 1.0 * atr_v
                tp_new = price + 2.0 * atr_v
                emoji = "🟢"
            else:
                pos_new = "SHORT"
                sl_new = price + 1.0 * atr_v
                tp_new = price - 2.0 * atr_v
                emoji = "🔴"

            fingerprint = f"{pos_new}:{round(price,2)}:{round(atr_v,2)}"
            if fingerprint == last_sent:
                print("Entrada ya avisada.")
                return

            msg = (
                f"{emoji} ABRIR {pos_new} ORO ({GOLD_TICKER}) 1H\n"
                f"Precio (cierre vela): {price}\n"
                f"EMA21: {ema21_v} | EMA50: {ema50_v}\n"
                f"MACD hist: {hist_v}\n"
                f"ATR(14): {atr_v:.2f}\n"
                f"SL sugerido: {sl_new:.2f}\n"
                f"TP sugerido: {tp_new:.2f}\n"
                f"👉 En eToro: SIN apalancamiento + coloca SL/TP al abrir."
            )
            send_telegram(msg)

            state["position"] = pos_new
            state["entry"] = float(round(price, 2))
            state["sl"] = float(round(sl_new, 2))
            state["tp"] = float(round(tp_new, 2))
            state["last_sent"] = fingerprint
            save_state(state)
            return

    # ===== DEBUG (si no hubo señal) =====
    if DEBUG:
        send_telegram(
            "DEBUG: No hay señal ahora.\n"
            f"position={position} entry={entry} sl={sl} tp={tp}\n"
            f"price={price} ema21={ema21_v} ema50={ema50_v} hist={hist_v} atr={atr_v}\n"
            f"dist_to_ema21={dist:.4f} max={MAX_DIST_TO_EMA21}\n"
            f"uptrend={uptrend} downtrend={downtrend} dxy_weak={dxy_weak} dxy_strong={dxy_strong}"
        )

    print("No signal.")

if __name__ == "__main__":
    main()
