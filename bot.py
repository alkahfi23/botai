from flask import Flask, request
import os, time, json, re
import pandas as pd, numpy as np
import requests
import ta
from datetime import datetime
import telebot
import gate_api
import websocket
import threading
from gate_api import Configuration, ApiClient, FuturesApi
from gate_api.exceptions import ApiException
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from chart_generator import draw_chart_by_timeframe

app = Flask(__name__)

TELEGRAM_BOT = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))
GATE_API_KEY = os.getenv("GATE_API_KEY")
GATE_API_SECRET = os.getenv("GATE_API_SECRET")

if not GATE_API_KEY or not GATE_API_SECRET:
    raise ValueError("❌ GATE_API_KEY atau GATE_API_SECRET tidak ditemukan di environment variable.")


# Konfigurasi dan klien
configuration = Configuration(
    key=GATE_API_KEY,
    secret=GATE_API_SECRET,
    host="https://api.gateio.ws"
)
api_client = ApiClient(configuration)

# ✅ Ini harus disini, sebelum fungsi-fungsi lain
futures_api = FuturesApi(api_client)

POPULAR_SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT", "ADA_USDT", "DOT_USDT", "AVAX_USDT", "DOGE_USDT", "MATIC_USDT"]

VALID_GATE_CONTRACTS = []

def get_all_gate_contracts(force_reload=False, max_retries=3):
    global VALID_GATE_CONTRACTS
    if not VALID_GATE_CONTRACTS or force_reload:
        url = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
        for attempt in range(max_retries):
            try:
                print(f"🔁 Ambil kontrak ke-{attempt + 1}...")
                resp = requests.get(url, timeout=10)
                if resp.status_code == 503:
                    raise Exception("503 Service Unavailable")
                resp.raise_for_status()
                VALID_GATE_CONTRACTS = [item["name"] for item in resp.json()]
                print("✅ Kontrak futures berhasil dimuat.")
                return VALID_GATE_CONTRACTS
            except Exception as e:
                print(f"❌ Gagal ambil kontrak (percobaan {attempt + 1}): {e}")
                time.sleep(2)

        print("⚠️ Gunakan fallback daftar kontrak manual.")
        VALID_GATE_CONTRACTS = [
            "BTC_USDT", "ETH_USDT", "BNB_USDT", "SOL_USDT", "XRP_USDT",
            "ADA_USDT", "DOGE_USDT", "DOT_USDT", "AVAX_USDT", "MATIC_USDT"
        ]
    return VALID_GATE_CONTRACTS

def normalize_symbol(symbol):
    contracts = get_all_gate_contracts()
    symbol = symbol.strip().upper()
    if symbol in contracts:
        return symbol
    if "_" not in symbol and symbol.endswith("USDT"):
        converted = symbol.replace("USDT", "_USDT")
        if converted in contracts:
            return converted
    return None

def escape_markdown(text):
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def get_klines(symbol="BTC_USDT", interval=None, intervals=None, contract_type="usdt", duration=20, verbose=True):
    if interval and not intervals:
        intervals = [interval]
    elif not intervals:
        intervals = ["1m", "5m", "15m"]  # Default intervals

    url = f"wss://fx-ws.gateio.ws/v4/ws/{contract_type}"
    klines_by_interval = {i: [] for i in intervals}

    def on_open(ws):
        for i in intervals:
            payload = {
                "time": int(time.time()),
                "channel": "futures.candlesticks",
                "event": "subscribe",
                "payload": [i, symbol]
            }
            ws.send(json.dumps(payload))
            if verbose:
                print(f"[OPEN] Subscribed to {symbol} @ {i}")

    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get("event") == "update" and data.get("channel") == "futures.candlesticks":
                kline_raw = data["result"][0]
                interval_raw = kline_raw["n"]  # already like '1m_BTC_USDT', but we subscribed to e.g. '1m'

                # Extract interval properly
                for i in intervals:
                    if interval_raw.startswith(i):
                        interval_used = i
                        break
                else:
                    return  # interval not matched

                kline_dict = {
                    "timestamp": pd.to_datetime(kline_raw['t'], unit="s"),
                    "open": float(kline_raw['o']),
                    "high": float(kline_raw['h']),
                    "low": float(kline_raw['l']),
                    "close": float(kline_raw['c']),
                    "volume": float(kline_raw['v']),
                    "amount": float(kline_raw['a']),
                }

                # Replace if same timestamp
                existing = klines_by_interval[interval_used]
                if existing and existing[-1]["timestamp"] == kline_dict["timestamp"]:
                    existing[-1] = kline_dict
                else:
                    existing.append(kline_dict)

                if verbose:
                    print(f"[KLINE] {symbol} @ {interval_used} → {kline_dict}")
        except Exception as e:
            if verbose:
                print("[PARSE ERROR]", e)

    def on_error(ws, error):
        print("[ERROR]", error)

    def on_close(ws, code, msg):
        if verbose:
            print("[CLOSED] WebSocket closed:", code, "-", msg)

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    t = threading.Thread(target=ws.run_forever)
    t.daemon = True
    t.start()

    time.sleep(duration)
    ws.close()
    t.join()

    dfs = {}
    for i, data in klines_by_interval.items():
        df = pd.DataFrame(data).sort_values("timestamp")
        dfs[i] = df

    return dfs
    
def get_24h_high_low(symbol):
    symbol = normalize_symbol(symbol)
    try:
        tickers = futures_api.list_futures_tickers(settle="usdt")
        t = next((x for x in tickers if x.name == symbol), None)
        if not t:
            return None, None
        return float(t.high_24h), float(t.low_24h)
    except:
        return None, None

def is_rsi_oversold(symbol, intervals="15m", limit=100):
    df = get_klines(symbol, [intervals], duration=20).get(intervals)
    if df is None or len(df) < 15: return False, None
    try:
        rsi = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi().iloc[-1]
        return rsi < 30, rsi
    except: return False, None

def check_rsi_overbought(symbols, intervals="15m", limit=100):
    result = []
    for s in symbols:
        df = get_klines(s, [intervals], duration=20).get(intervals)
        if df is None or len(df) < 15: continue
        try:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            if rsi > 70: result.append((s, round(rsi, 2)))
        except: continue
    return sorted(result, key=lambda x: -x[1])

def detect_reversal_candle(df):
    if len(df) < 3: return None
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    body = lambda c: abs(c['close'] - c['open'])
    upper = lambda c: c['high'] - max(c['close'], c['open'])
    lower = lambda c: min(c['close'], c['open']) - c['low']
    is_bullish = lambda c: c['close'] > c['open']
    is_bearish = lambda c: c['close'] < c['open']
    body_ratio = lambda c: body(c) / (c['high'] - c['low'] + 1e-6)
    if body_ratio(c2) < 0.3 and lower(c2) > 2 * body(c2) and is_bullish(c3): return "Hammer"
    if body_ratio(c2) < 0.3 and upper(c2) > 2 * body(c2) and is_bullish(c3): return "InvertedHammer"
    if is_bearish(c1) and is_bullish(c2) and c2['close'] > c1['open'] and is_bullish(c3): return "Engulfing"
    if body_ratio(c2) < 0.3 and upper(c2) > 2 * body(c2) and is_bearish(c3): return "ShootingStar"
    if is_bullish(c1) and is_bearish(c2) and c2['close'] < c1['open'] and is_bearish(c3): return "Engulfing"
    return None

def analyze_multi_timeframe(symbol):
    df_15m = get_klines(symbol, '15m', 500)
    if df_15m is None:
        print(f"[DEBUG] ❌ df_15m kosong untuk {symbol}")
    df_5m = get_klines(symbol, '5m', 500)
    if df_5m is None:
        print(f"[DEBUG] ❌ df_5m kosong untuk {symbol}")
    df_1m = get_klines(symbol, '1m', 500)
    if df_1m is None:
        print(f"[DEBUG] ❌ df_1m kosong untuk {symbol}")

    if not all([df_15m is not None, df_5m is not None, df_1m is not None]):
        return f"❌ Gagal ambil data {symbol}", "ERROR", 0

    try:
        for df in [df_15m, df_5m, df_1m]:
            if len(df) < 20:
                return f"❌ Data terlalu pendek untuk analisa {symbol}", "ERROR", 0

            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_H'] = bb.bollinger_hband()
            df['BB_L'] = bb.bollinger_lband()

        if df_1m.empty or len(df_1m) < 1:
            return f"❌ Data 1m kosong untuk {symbol}", "ERROR", 0

        last = df_1m.iloc[-1]

        candle_pattern = detect_reversal_candle(df_1m)

        trend_15m = "UP" if df_15m['close'].iloc[-1] > df_15m['EMA20'].iloc[-1] else "DOWN"
        trend_5m = "UP" if df_5m['close'].iloc[-1] > df_5m['EMA20'].iloc[-1] else "DOWN"

        high_24h, low_24h = get_24h_high_low(symbol)
        if high_24h is None or low_24h is None:
            high_24h, low_24h = last['close'], last['close']

        near_low = last['close'] <= low_24h * 1.01
        near_high = last['close'] >= high_24h * 0.99

        signal, entry, stop_loss, take_profit = None, None, None, None

        if trend_15m == "UP" and trend_5m == "UP":
            if last['RSI'] < 30 and last['close'] < last['BB_L'] and near_low and candle_pattern in ['Hammer', 'InvertedHammer', 'Engulfing']:
                signal = "LONG"
                entry = last['close']
                sl = df_1m[df_1m['close'] < df_1m['BB_L']]['low']
                stop_loss = sl.iloc[-1] if not sl.empty else df_1m['low'].min()
                take_profit = entry + 2 * (entry - stop_loss)

        elif trend_15m == "DOWN" and trend_5m == "DOWN":
            if last['RSI'] > 70 and last['close'] > df_1m['BB_H'].iloc[-1] and near_high and candle_pattern in ['ShootingStar', 'Engulfing']:
                signal = "SHORT"
                entry = last['close']
                sl = df_1m[df_1m['close'] > df_1m['BB_H']]['high']
                stop_loss = sl.iloc[-1] if not sl.empty else df_1m['high'].max()
                take_profit = entry - 2 * (stop_loss - entry)

        msg = f"""📉 Pair: {symbol}
Trend 15m: {trend_15m} | 5m: {trend_5m}
RSI 1m: {last['RSI']:.2f} | Price: {last['close']:.2f}
Candle: `{candle_pattern or 'N/A'}`
24H High: {high_24h:.2f} | Low: {low_24h:.2f}
{"⚠️ Near LOW 24H" if near_low else ""}
{"⚠️ Near HIGH 24H" if near_high else ""}
"""

        if signal:
            msg += f"""✅ Sinyal: {signal}
🎯 Entry: {entry:.2f}
🛑 SL: {stop_loss:.2f}
🎯 TP: {take_profit:.2f}"""
        else:
            msg += "🚫 Tidak ada sinyal valid saat ini."

        return msg, signal or "NONE", entry or 0

    except Exception as e:
        return f"❌ Error analisa {symbol}: {e}", "ERROR", 0


# === WEBHOOK ===
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)

        if not data:
            print("❌ Tidak ada data masuk ke webhook")
            return "NO DATA", 400

        # === HANDLE TELEGRAM MESSAGE ===
        if "message" in data:
            return handle_telegram_message(data["message"])

        # === HANDLE GATE.IO WEBHOOK ===
        return handle_gateio_webhook(data)

    except Exception as e:
        print(f"❌ Error di webhook: {e}")
        return "ERROR", 500


def handle_telegram_message(message):
    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip().upper()
    print(f"📩 Pesan masuk dari user: {text}")

    if text == "/HELP":
        return send_help_message(chat_id)

    elif text in ["RSI", "RSIS"]:
        return process_rsi_command(chat_id, text)

    elif text.startswith("CHART "):
        return process_chart_command(chat_id, text)

    elif text in ["LONG", "SHORT"]:
        return find_signals(chat_id, text)

    else:
        return analyze_single_pair(chat_id, text)


def send_help_message(chat_id):
    markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Backtest", callback_data="BACKTEST")],
        [InlineKeyboardButton("✅ Cari LONG", callback_data="LONG"),
         InlineKeyboardButton("⛔ Cari SHORT", callback_data="SHORT")]
    ])
    help_msg = (
        "🤖 *Bot Trading Gate.io:*\n"
        "📈 /BACKTEST — Backtest semua pair\n"
        "📉 RSI — Cari coin dengan RSI < 30\n"
        "📈 RSIS — Cari coin RSI > 70\n"
        "🕯️ CHART BTCUSDT — Analisa chart coin\n"
        "🛒 BTCUSDT — Langsung analisa pair"
    )
    TELEGRAM_BOT.send_message(chat_id, escape_markdown(help_msg), parse_mode="MarkdownV2", reply_markup=markup)
    return "OK"


def process_rsi_command(chat_id, command):
    TELEGRAM_BOT.send_message(chat_id, "🔎 Memproses data RSI...")

    result = []
    if command == "RSI":
        for sym in POPULAR_SYMBOLS:
            ok, val = is_rsi_oversold(sym)
            if ok:
                result.append(f"🔻 *{sym}* - RSI `{val:.2f}`")
        msg = "\n".join(result) if result else "✅ Tidak ada coin RSI < 30"
    else:
        result = check_rsi_overbought(POPULAR_SYMBOLS)
    if isinstance(result, list):  # ✅ pastikan iterable
        msg = "*Overbought 15m:*\n" + "\n".join(f"{s} - RSI {r}" for s, r in result) if result else "✅ Tidak ada RSI > 70"
    else:
        msg = f"❌ Format return tidak valid: {result}"
        
    TELEGRAM_BOT.send_message(chat_id, escape_markdown(msg), parse_mode="MarkdownV2")
    return "OK"


def process_chart_command(chat_id, text):
    symbol = normalize_symbol(text.split()[1])
    if not symbol:
        TELEGRAM_BOT.send_message(chat_id, escape_markdown("❌ Simbol tidak dikenali."), parse_mode="MarkdownV2")
        return "OK"

    msg, _, _ = analyze_multi_timeframe(symbol)
    TELEGRAM_BOT.send_message(chat_id, escape_markdown(msg), parse_mode="MarkdownV2")

    chart = draw_chart_by_timeframe(symbol, "1m")
    if chart:
        TELEGRAM_BOT.send_photo(chat_id, chart)
    return "OK"


def find_signals(chat_id, signal_type):
    found = False
    for symbol in POPULAR_SYMBOLS:
        try:
            msg, signal, _ = analyze_multi_timeframe(symbol)
            if signal == signal_type:
                TELEGRAM_BOT.send_message(chat_id, escape_markdown(msg), parse_mode="MarkdownV2")
                found = True
        except Exception as e:
            print(f"[{symbol}] ❌ Error analisa: {e}")
    if not found:
        TELEGRAM_BOT.send_message(chat_id, escape_markdown(f"🚫 Tidak ditemukan sinyal {signal_type}"), parse_mode="MarkdownV2")
    return "OK"


def analyze_single_pair(chat_id, text):
    symbol = normalize_symbol(text)
    if symbol:
        msg, signal, _ = analyze_multi_timeframe(symbol)
        TELEGRAM_BOT.send_message(chat_id, escape_markdown(msg), parse_mode="MarkdownV2")

        chart = draw_chart_by_timeframe(symbol, "1m")
        if chart:
            TELEGRAM_BOT.send_photo(chat_id, chart)
    else:
        TELEGRAM_BOT.send_message(chat_id, escape_markdown("❌ Simbol tidak valid."), parse_mode="MarkdownV2")
    return "OK"


def handle_gateio_webhook(data):
    print(f"📡 Webhook Gate.io masuk:\n{data}")

    symbol = data.get("symbol") or data.get("currency_pair")
    if not symbol:
        return "NO SYMBOL", 400

    try:
        is_oversold, rsi_val = is_rsi_oversold(symbol)
        if is_oversold:
            print(f"[RSI] 🔻 {symbol} RSI oversold: {rsi_val:.2f}")
        else:
            print(f"[RSI] {symbol} aman, RSI: {rsi_val:.2f}")
        return "GATEIO OK", 200
    except Exception as e:
        print(f"[RSI ERROR] {symbol}: {e}")
        return "GATEIO ERROR", 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
