from flask import Flask, request
import os, time, json, re
import pandas as pd, numpy as np
import requests
import ta
from datetime import datetime
import telebot
import gate_api
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
    
# Technicals

def get_klines(symbol, interval="1m", limit=100, max_retries=3):
    symbol = normalize_symbol(symbol)
    if not symbol:
        print(f"❌ Symbol tidak valid: {symbol}")
        return None

    for attempt in range(1, max_retries + 1):
        try:
            candles = futures_api.list_futures_candlesticks(
                settle="usdt",
                contract=symbol,
                interval=interval,
                limit=limit
            )
            if not candles or len(candles) < 5:
                print(f"⚠️ Data candlestick terlalu sedikit: {symbol} ({len(candles)} baris)")
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'volume', 'close', 'high', 'low', 'open'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()

        except gate_api.exceptions.ApiException as e:
            print(f"❌ APIException ambil klines {symbol} (percobaan {attempt}): {e.status} - {e.body}")
            if e.status == 503 and attempt < max_retries:
                time.sleep(2 * attempt)  # Exponential backoff
            else:
                break
        except Exception as e:
            print(f"❌ Error umum ambil klines {symbol}: {e}")
            break

    return None


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

def is_rsi_oversold(symbol, interval="15m", limit=100):
    df = get_klines(symbol, interval, limit)
    if df is None or len(df) < 15: return False, None
    try:
        rsi = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi().iloc[-1]
        return rsi < 30, rsi
    except: return False, None

def check_rsi_overbought(symbols, interval="15m", limit=100):
    result = []
    for s in symbols:
        df = get_klines(s, interval, limit)
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
    df_5m = get_klines(symbol, '5m', 500)
    df_1m = get_klines(symbol, '1m', 500)
    if not all([df_15m is not None, df_5m is not None, df_1m is not None]):
        return f"❌ Gagal ambil data {symbol}", "ERROR", 0
    try:
        for df in [df_15m, df_5m, df_1m]:
            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_H'] = bb.bollinger_hband(); df['BB_L'] = bb.bollinger_lband()
    except Exception as e:
        return f"❌ Error indikator {symbol}: {e}", "ERROR", 0
    last = df_1m.iloc[-1]
    candle_pattern = detect_reversal_candle(df_1m)
    trend_15m = "UP" if df_15m['close'].iloc[-1] > df_15m['EMA20'].iloc[-1] else "DOWN"
    trend_5m = "UP" if df_5m['close'].iloc[-1] > df_5m['EMA20'].iloc[-1] else "DOWN"
    high_24h, low_24h = get_24h_high_low(symbol)
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
        if last['RSI'] > 70 and last['close'] > last['BB_H'] and near_high and candle_pattern in ['ShootingStar', 'Engulfing']:
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

# === WEBHOOK ===

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        if not data:
            return "NO DATA"

        if "message" in data:
            chat_id = data["message"]["chat"]["id"]
            text = data["message"].get("text", "").strip().upper()
            print(f"📩 User input: {text}")

            if text == "/HELP":
                markup = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔁 Backtest", callback_data="BACKTEST")],
                    [InlineKeyboardButton("✅ Cari LONG", callback_data="LONG"), InlineKeyboardButton("⛔ Cari SHORT", callback_data="SHORT")]
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

            elif text in ["RSI", "RSIS"]:
                TELEGRAM_BOT.send_message(chat_id, "🔎 Memproses data RSI...")
                if text == "RSI":
                    result = []
                    for sym in POPULAR_SYMBOLS:
                        ok, val = is_rsi_oversold(sym)
                        if ok:
                            result.append(f"🔻 *{sym}* - RSI `{val:.2f}`")
                    msg = "\n".join(result) if result else "✅ Tidak ada coin RSI < 30"
                else:
                    result = check_rsi_overbought(POPULAR_SYMBOLS)
                    msg = "*Overbought 15m:*\n" + "\n".join(f"{s} - RSI {r}" for s, r in result) if result else "✅ Tidak ada RSI > 70"
                TELEGRAM_BOT.send_message(chat_id, escape_markdown(msg), parse_mode="MarkdownV2")

            elif text.startswith("CHART "):
                symbol = normalize_symbol(text.split()[1])
                if not symbol:
                    TELEGRAM_BOT.send_message(chat_id, escape_markdown("❌ Simbol tidak dikenali."), parse_mode="MarkdownV2")
                    return "OK"
                msg, _, _ = analyze_multi_timeframe(symbol)
                TELEGRAM_BOT.send_message(chat_id, escape_markdown(msg), parse_mode="MarkdownV2")
                chart = draw_chart_by_timeframe(symbol, "1m")
                if chart:
                    TELEGRAM_BOT.send_photo(chat_id, chart)

            elif text in ["LONG", "SHORT"]:
                found = False
                for s in POPULAR_SYMBOLS:
                    try:
                        msg, signal, _ = analyze_multi_timeframe(s)
                        if signal == text:
                            TELEGRAM_BOT.send_message(chat_id, escape_markdown(msg), parse_mode="MarkdownV2")
                            found = True
                    except Exception as e:
                        print(f"{s}: {e}")
                if not found:
                    TELEGRAM_BOT.send_message(chat_id, escape_markdown(f"🚫 Tidak ditemukan sinyal {text}"), parse_mode="MarkdownV2")

            else:
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

    except Exception as e:
        print(f"❌ Error di webhook: {e}")
        return "ERROR", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
