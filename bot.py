from flask import Flask, request
import os
import pandas as pd
import numpy as np
import ta
import telebot
from datetime import datetime
from binance.client import Client
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from ta.momentum import RSIIndicator
from chart_generator import draw_chart_by_timeframe  # Pastikan file ini tersedia dan berfungsi

app = Flask(__name__)

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_BOT = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

POPULAR_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT"
]

def get_klines(symbol, interval="5m", limit=100):
    try:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not raw or len(raw) < limit // 2:
            print(f"⚠️ Data kline {symbol}-{interval} tidak mencukupi. Dapat: {len(raw)}")
            return None

        df = pd.DataFrame(raw, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Ubah kolom angka jadi float, dengan error handling
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ ERROR get_klines({symbol}, {interval}): {e}")
        return None

def get_24h_high_low(symbol):
    try:
        ticker = client.get_ticker(symbol=symbol)
        high = float(ticker['highPrice'])
        low = float(ticker['lowPrice'])
        return high, low
    except Exception as e:
        print(f"❌ Gagal ambil 24h high/low untuk {symbol}: {e}")
        return None, None

def is_rsi_oversold(symbol, interval="15m", limit=100):
    df = get_klines(symbol, interval, limit)
    if df is None or df.empty or len(df) < 15:
        return False, None

    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=df["close"], window=14)
        df["RSI_14"] = rsi_indicator.rsi()
        latest_rsi = df["RSI_14"].iloc[-1]
        return latest_rsi < 30, latest_rsi
    except Exception as e:
        print(f"❌ Error hitung RSI {symbol}: {e}")
        return False, None
        
def check_rsi_overbought(symbols, interval="15m", limit=100):
    overbought_list = []
    for symbol in symbols:
        df = get_klines(symbol, interval, limit)
        if df is None or len(df) < 15:
            continue
        try:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            if rsi > 70:
                overbought_list.append((symbol, round(rsi, 2)))
        except Exception as e:
            print(f"❌ Error RSI {symbol}: {e}")
    return sorted(overbought_list, key=lambda x: -x[1])  # Urutkan dari RSI tertinggi

# Ganti fungsi ini
def detect_reversal_candle(df):
    if len(df) < 3:
        return None

    c1 = df.iloc[-3]  # Candle pertama (pola)
    c2 = df.iloc[-2]  # Candle kedua
    c3 = df.iloc[-1]  # Candle ketiga (konfirmasi)

    def body(c): return abs(c['close'] - c['open'])
    def upper(c): return c['high'] - max(c['close'], c['open'])
    def lower(c): return min(c['close'], c['open']) - c['low']
    def is_bullish(c): return c['close'] > c['open']
    def is_bearish(c): return c['close'] < c['open']
    def body_ratio(c): return body(c) / (c['high'] - c['low'] + 1e-6)

    # --- HAMMER ---
    if body_ratio(c2) < 0.3 and lower(c2) > 2 * body(c2) and upper(c2) < body(c2):
        if is_bullish(c3):  # konfirmasi naik
            return "Hammer"

    # --- INVERTED HAMMER ---
    if body_ratio(c2) < 0.3 and upper(c2) > 2 * body(c2) and lower(c2) < body(c2):
        if is_bullish(c3):
            return "InvertedHammer"

    # --- BULLISH ENGULFING ---
    if is_bearish(c1) and is_bullish(c2) and c2['open'] < c1['close'] and c2['close'] > c1['open']:
        if is_bullish(c3):
            return "Engulfing"

    # --- SHOOTING STAR ---
    if body_ratio(c2) < 0.3 and upper(c2) > 2 * body(c2) and lower(c2) < body(c2):
        if is_bearish(c3):
            return "ShootingStar"

    # --- BEARISH ENGULFING ---
    if is_bullish(c1) and is_bearish(c2) and c2['open'] > c1['close'] and c2['close'] < c1['open']:
        if is_bearish(c3):
            return "Engulfing"

    return None

def backtest_strategy(symbol, interval="1m", limit=500):
    df = get_klines(symbol, interval, limit)
    if df is None or df.shape[0] < 100:
        return []

    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_H'] = bb.bollinger_hband()
    df['BB_L'] = bb.bollinger_lband()

    results = []
    for i in range(30, len(df) - 10):
        candle = df.iloc[i]
        trend = "UP" if df.iloc[i-5:i]['close'].mean() > df.iloc[i-5:i]['EMA20'].mean() else "DOWN"
        candle_pattern = detect_reversal_candle(df.iloc[i-2:i+1])

        entry = candle['close']
        result = None
        take_profit = None
        stop_loss = None
        RR = None

        if trend == "UP" and candle['RSI'] < 30 and candle['close'] < candle['BB_L'] and candle_pattern in ['Hammer', 'InvertedHammer', 'Engulfing']:
            stop_loss = df.iloc[i]['low']
            take_profit = entry + (entry - stop_loss) * 2
            future = df.iloc[i+1:i+6]
            if (future['high'] >= take_profit).any():
                result = "WIN"
            elif (future['low'] <= stop_loss).any():
                result = "LOSS"

        elif trend == "DOWN" and candle['RSI'] > 70 and candle['close'] > candle['BB_H'] and candle_pattern in ['ShootingStar', 'Engulfing']:
            stop_loss = df.iloc[i]['high']
            take_profit = entry - (stop_loss - entry) * 2
            future = df.iloc[i+1:i+6]
            if (future['low'] <= take_profit).any():
                result = "WIN"
            elif (future['high'] >= stop_loss).any():
                result = "LOSS"

        if result:
            RR = 2.0
            results.append({"index": i, "result": result, "RR": RR})

    return results

def backtest_all_symbols(symbols, interval="1m", limit=500):
    summary = []
    for symbol in symbols:
        results = backtest_strategy(symbol, interval, limit)
        if not results:
            continue
        total = len(results)
        wins = sum(1 for r in results if r["result"] == "WIN")
        losses = sum(1 for r in results if r["result"] == "LOSS")
        rr_list = [r["RR"] for r in results if r["RR"] is not None]
        accuracy = (wins / total) * 100 if total > 0 else 0
        avg_rr = np.mean(rr_list) if rr_list else 0
        profit_factor = (wins * 2) / (losses * 1) if losses > 0 else float("inf")
        summary.append({
            "symbol": symbol,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "accuracy": round(accuracy, 2),
            "avg_rr": round(avg_rr, 2),
            "profit_factor": round(profit_factor, 2) if isinstance(profit_factor, float) else "∞"
        })
    return summary

def format_summary(summary):
    lines = ["📊 *Rangkuman Backtest Semua Pair:*\n"]
    lines.append("Pair | Trade | Win | Loss | Akurasi | R:R | Profit")
    lines.append("-" * 45)
    for s in summary:
        lines.append(f"{s['symbol']} | {s['total_trades']} | {s['wins']} | {s['losses']} | {s['accuracy']}% | {s['avg_rr']} | {s['profit_factor']}")
    return "\n".join(lines)

def analyze_multi_timeframe(symbol):
    df_15m = get_klines(symbol, '15m', 500)
    df_5m = get_klines(symbol, '5m', 500)
    df_1m = get_klines(symbol, '1m', 500)

    if df_1m is None or df_5m is None or df_15m is None:
        print(f"⚠️ Gagal ambil data untuk {symbol}. Timeframe yang error:")
        if df_15m is None: print("- 15m")
        if df_5m is None: print("- 5m")
        if df_1m is None: print("- 1m")
        return f"❌ Gagal ambil data {symbol}", "ERROR", 0

    try:
        for df in [df_15m, df_5m, df_1m]:
            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_H'] = bb.bollinger_hband()
            df['BB_L'] = bb.bollinger_lband()
    except Exception as e:
        print(f"❌ Error hitung indikator: {e}")
        return f"❌ Error indikator {symbol}: {e}", "ERROR", 0

    signal = None
    entry = None
    stop_loss = None
    take_profit = None
    current_price = df_1m['close'].iloc[-1]
    candle_pattern = detect_reversal_candle(df_1m)

    trend_15m = "UP" if df_15m['close'].iloc[-1] > df_15m['EMA20'].iloc[-1] else "DOWN"
    trend_5m = "UP" if df_5m['close'].iloc[-1] > df_5m['EMA20'].iloc[-1] else "DOWN"
    last = df_1m.iloc[-1]

    # Ambil high/low 24 jam
    high_24h, low_24h = get_24h_high_low(symbol)
    if high_24h is None or low_24h is None:
        return f"❌ Gagal ambil data 24H untuk {symbol}", "ERROR", 0

    is_near_24h_low = current_price <= (low_24h + 0.01 * low_24h)
    is_near_24h_high = current_price >= (high_24h - 0.01 * high_24h)

    if trend_15m == "UP" and trend_5m == "UP":
        if last['RSI'] < 30 and last['close'] < last['BB_L'] and is_near_24h_low and candle_pattern in ['Hammer', 'InvertedHammer', 'Engulfing']:
            signal = "LONG"
            entry = current_price
            prev_below_bb = df_1m[:-1][df_1m['close'] < df_1m['BB_L']]
            stop_loss = prev_below_bb['low'].iloc[-1] if not prev_below_bb.empty else df_1m['low'].min()
            risk = entry - stop_loss
            take_profit = entry + (2 * risk)

    elif trend_15m == "DOWN" and trend_5m == "DOWN":
        if last['RSI'] > 70 and last['close'] > last['BB_H'] and is_near_24h_high and candle_pattern in ['ShootingStar', 'Engulfing']:
            signal = "SHORT"
            entry = current_price
            prev_above_bb = df_1m[:-1][df_1m['close'] > df_1m['BB_H']]
            stop_loss = prev_above_bb['high'].iloc[-1] if not prev_above_bb.empty else df_1m['high'].max()
            risk = stop_loss - entry
            take_profit = entry - (2 * risk)

    result = f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n"
    result += f"📉 Pair: {symbol}\n"
    result += f"Trend 15m: {trend_15m}\n"
    result += f"Trend 5m: {trend_5m}\n"
    result += f"🕯️ RSI 1m: {last['RSI']:.2f}\n"
    result += f"📊 Harga Sekarang: {current_price:.2f}\n"
    result += f"🕯️ Pola Candle Terbaca: `{candle_pattern or 'Tidak ada'}`\n"
    result += f"📈 High 24H: {high_24h:.2f}\n"
    result += f"📉 Low 24H: {low_24h:.2f}\n"

    if is_near_24h_low:
        result += "⚠️ Dekat dengan **LOW 24H** (potensi rebound)\n"
    if is_near_24h_high:
        result += "⚠️ Dekat dengan **HIGH 24H** (potensi koreksi)\n"

    if signal:
        result += f"\n✅ Sinyal Terdeteksi: {signal}\n"
        result += f"🎯 Entry: {entry:.2f}\n"
        result += f"🛑 Stop Loss: {stop_loss:.2f}\n"
        result += f"🎯 Take Profit: {take_profit:.2f}\n"
    else:
        result += "\n🚫 Tidak ada sinyal valid saat ini."

    return result, signal or "NONE", entry or 0



@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()

    # === Handle callback queries (inline button clicks) ===
    if "callback_query" in data:
        callback_data = data["callback_query"]["data"]
        chat_id = data["callback_query"]["message"]["chat"]["id"]

        if callback_data == "BACKTEST":
            TELEGRAM_BOT.send_message(chat_id, "🧪 Memulai backtest semua simbol...")
            summary = backtest_all_symbols(POPULAR_SYMBOLS, interval="1m", limit=500)
            formatted = format_summary(summary)
            TELEGRAM_BOT.send_message(chat_id, formatted, parse_mode="Markdown")
            return "OK"

        if callback_data in ["LONG", "SHORT"]:
            found = False
            TELEGRAM_BOT.send_message(chat_id, f"🔍 Mencari sinyal `{callback_data}` di 10 coin populer...", parse_mode="Markdown")
            for symbol in POPULAR_SYMBOLS:
                try:
                    message, signal, entry = analyze_multi_timeframe(symbol)
                    if signal == callback_data:
                        TELEGRAM_BOT.send_message(chat_id, message, parse_mode="Markdown")
                        chart = draw_chart_by_timeframe(symbol, "1m")
                        if chart:
                            TELEGRAM_BOT.send_photo(chat_id=chat_id, photo=chart)

                        markup = InlineKeyboardMarkup()
                        button = InlineKeyboardButton(
                            text=f"Buka {symbol} di Binance 📲",
                            url=f"https://www.binance.com/en/futures/{symbol}?ref=GRO_16987_24H8Y"
                        )
                        markup.add(button)
                        TELEGRAM_BOT.send_message(chat_id, "Klik tombol di bawah untuk buka di aplikasi Binance:", reply_markup=markup)
                        found = True
                except Exception as e:
                    print(f"Error cek {symbol}: {e}")

            if not found:
                TELEGRAM_BOT.send_message(chat_id, f"❌ Tidak ditemukan sinyal `{callback_data}` saat ini.", parse_mode="Markdown")
            return "OK"

        if callback_data.startswith("CHART_"):
            try:
                _, symbol, timeframe = callback_data.split("_")
                chart = draw_chart_by_timeframe(symbol, timeframe)
                caption = f"📊 {symbol} - {timeframe.upper()} Chart"
                TELEGRAM_BOT.send_photo(chat_id=chat_id, photo=chart, caption=caption)

                markup = InlineKeyboardMarkup()
                btn_binance = InlineKeyboardButton(
                    text=f"Buka {symbol} di Binance 📲",
                    url=f"https://www.binance.com/en/futures/{symbol}?ref=GRO_16987_24H8Y"
                )
                markup.add(btn_binance)
                TELEGRAM_BOT.send_message(chat_id, "Klik tombol di bawah untuk buka di aplikasi Binance:", reply_markup=markup)
            except Exception as e:
                TELEGRAM_BOT.send_message(chat_id, f"⚠️ Gagal generate chart: {e}")
            return "OK"

    # === Handle regular messages ===
    if "message" in data and "text" in data["message"]:
        text = data["message"]["text"].strip().upper()
        chat_id = data["message"]["chat"]["id"]

        # === HELP ===
        if text == "/HELP":
            help_text = (
                "🤖 *Panduan Bot Signal Trading:*\n\n"
                "🔍 Kirim salah satu perintah berikut:\n"
                "/BACKTEST — Jalankan backtest semua pair populer\n"
                "LONG — Cari sinyal BUY (naik)\n"
                "SHORT — Cari sinyal SELL (turun)\n"
                "RSI — Tampilkan coin dengan RSI Oversold (15m)\n"
                "RSIS — Tampilkan coin dengan RSI > 70 (Overbought)\n"
                "CHART BTCUSDT — Lihat chart + sinyal untuk pair tertentu\n"
                "BTCUSDT, ETHUSDT, dst — Analisa spesifik pair\n"
                "/HELP — Tampilkan bantuan ini\n\n"
                "💡 Tips: Gunakan di saat volatilitas tinggi untuk sinyal terbaik."
            )

            markup = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🔁 Backtest", callback_data="BACKTEST"),
                    InlineKeyboardButton("✅ Cari LONG", callback_data="LONG"),
                    InlineKeyboardButton("⛔ Cari SHORT", callback_data="SHORT")
                ]
            ])

            TELEGRAM_BOT.send_message(chat_id, help_text, parse_mode="Markdown", reply_markup=markup)
            return "OK"
            
         # === RSI Overbought ===
        elif text == "RSIS":
            TELEGRAM_BOT.send_message(chat_id, "📈 Mengecek RSI Overbought di 15m timeframe...")
            result = check_rsi_overbought(POPULAR_SYMBOLS, interval="15m")
            if not result:
                TELEGRAM_BOT.send_message(chat_id, "⚠️ Tidak ditemukan coin dengan RSI > 70 saat ini.")
            else:
                msg = "*📊 RSI Overbought 15m:*\n\n"
                msg += "Pair | RSI\n"
                msg += "-" * 15 + "\n"
                for sym, rsi in result:
                    msg += f"{sym} | {rsi}\n"
                TELEGRAM_BOT.send_message(chat_id, msg, parse_mode="Markdown")

        # === RSI Oversold ===
        if text == "RSI":
            TELEGRAM_BOT.send_message(chat_id, "📉 Mendeteksi RSI oversold pada coin populer (15m)...")
            oversold_list = []

            for symbol in POPULAR_SYMBOLS:
                try:
                    is_oversold, rsi_val = is_rsi_oversold(symbol, interval="15m")
                    if is_oversold:
                        oversold_list.append(f"🔻 *{symbol}* - RSI: `{rsi_val:.2f}`")

                        chart = draw_chart_by_timeframe(symbol, "15m")
                        if chart:
                            TELEGRAM_BOT.send_photo(chat_id=chat_id, photo=chart, caption=f"{symbol} - RSI: {rsi_val:.2f}")
                except Exception as e:
                    print(f"Error cek RSI {symbol}: {e}")

            if oversold_list:
                reply = "*Coin dengan RSI Oversold (15m)*:\n\n" + "\n".join(oversold_list)
            else:
                reply = "✅ Tidak ada coin dengan RSI < 30 di timeframe 15m saat ini."
            TELEGRAM_BOT.send_message(chat_id, reply, parse_mode="Markdown")
            return "OK"

        # === CHART SYMBOL ===
        if text.startswith("CHART "):
            parts = text.split()
            if len(parts) == 2:
                symbol = parts[1]
                try:
                    message, signal, entry = analyze_multi_timeframe(symbol)
                    TELEGRAM_BOT.send_message(chat_id, message, parse_mode="Markdown")

                    markup = InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("1 Menit", callback_data=f"CHART_{symbol}_1m"),
                            InlineKeyboardButton("5 Menit", callback_data=f"CHART_{symbol}_5m"),
                        ],
                        [
                            InlineKeyboardButton("15 Menit", callback_data=f"CHART_{symbol}_15m"),
                            InlineKeyboardButton("1 Jam", callback_data=f"CHART_{symbol}_1h"),
                        ]
                    ])
                    TELEGRAM_BOT.send_message(chat_id, f"Pilih timeframe untuk {symbol}:", reply_markup=markup)
                except Exception as e:
                    TELEGRAM_BOT.send_message(chat_id, f"⚠️ Gagal mengambil chart: {e}")
            else:
                TELEGRAM_BOT.send_message(chat_id, "⚠️ Format tidak valid. Contoh: `CHART BTCUSDT`", parse_mode="Markdown")
            return "OK"

        # === Simbol langsung ===
        if len(text) >= 6 and text.isalnum():
            try:
                message, signal, entry = analyze_multi_timeframe(text)
                TELEGRAM_BOT.send_message(chat_id, message, parse_mode="Markdown")

                if signal != "NONE":
                    chart = draw_chart_by_timeframe(text, "1m")
                    if chart:
                        TELEGRAM_BOT.send_photo(chat_id, chart)

                    markup = InlineKeyboardMarkup()
                    button = InlineKeyboardButton(
                        text=f"Buka {text} di Binance 📲",
                        url=f"https://www.binance.com/en/futures/{text}?ref=GRO_16987_24H8Y"
                    )
                    markup.add(button)
                    TELEGRAM_BOT.send_message(chat_id, "Klik tombol di bawah untuk buka di aplikasi Binance:", reply_markup=markup)
            except Exception as e:
                TELEGRAM_BOT.send_message(chat_id, f"⚠️ Error analisis: {e}")
            return "OK"

        TELEGRAM_BOT.send_message(chat_id, "⚠️ Format simbol tidak valid atau terlalu pendek.")
    return "OK"

   
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
