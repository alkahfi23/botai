import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from scipy.signal import argrelextrema
import ta
import requests
import logging
import matplotlib.dates as mdates
import gate_api
import time
import websocket
import json
import threading
from gate_api import Configuration, ApiClient, FuturesApi
from gate_api.exceptions import ApiException
from ta.momentum import RSIIndicator
from ta.trend import MACD


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

logging.basicConfig(level=logging.INFO)

# Jangan inisialisasi di luar fungsi jika startup internet lambat
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
    
def get_klines(symbol="BTC_USDT", interval=None, intervals=None, contract_type="usdt", duration=20):
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
            print(f"[OPEN] Subscribed to {symbol} @ {i}")

    def on_message(ws, message):
        data = json.loads(message)
        if data.get("event") == "update" and data.get("channel") == "futures.candlesticks":
            kline_raw = data["result"][0]
            interval_raw = kline_raw["n"].split("_")[0]
            kline_dict = {
                "timestamp": pd.to_datetime(kline_raw['t'], unit="s"),
                "open": float(kline_raw['o']),
                "high": float(kline_raw['h']),
                "low": float(kline_raw['l']),
                "close": float(kline_raw['c']),
                "volume": float(kline_raw['v']),
                "amount": float(kline_raw['a']),
            }
            if interval_raw in klines_by_interval:
                klines_by_interval[interval_raw].append(kline_dict)
                print(f"[KLINE] {symbol} @ {interval_raw} → {kline_dict}")

    def on_error(ws, error):
        print("[ERROR]", error)

    def on_close(ws, code, msg):
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

    
def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = [True] * len(df)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upperband.iloc[i - 1]:
            supertrend[i] = True
        elif df['close'].iloc[i] < lowerband.iloc[i - 1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i - 1]
            if supertrend[i] and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if not supertrend[i] and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]
    return pd.DataFrame({
        'supertrend': supertrend,
        'upperband': upperband,
        'lowerband': lowerband
    }, index=df.index)

def draw_chart_by_timeframe(symbol='BTC_USDT', tf='1m'):
    symbol = normalize_symbol(symbol)
    df = get_klines(symbol, interval=tf)
    if df is None or len(df) < 100:
        return None
    df['EMA50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
    df['EMA200'] = ta.trend.EMAIndicator(df['close'], 200).ema_indicator()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close']).macd()
    macd_signal = MACD(df['close']).macd_signal()
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    st = calculate_supertrend(df)
    df['Volume_MA20'] = df['volume'].rolling(20).mean()

    df_ohlc = df[['open', 'high', 'low', 'close']].copy()
    df_ohlc['Date'] = df_ohlc.index.map(mdates.date2num)
    ohlc = df_ohlc[['Date', 'open', 'high', 'low', 'close']]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1.5, 1]})

    from mplfinance.original_flavor import candlestick_ohlc
    candlestick_ohlc(ax1, ohlc.values, width=0.0005, colorup='g', colordown='r', alpha=0.8)
    ax1.plot(df.index, df['EMA50'], color='lime', label='EMA50')
    ax1.plot(df.index, df['EMA200'], color='orange', label='EMA200')
    ax1.plot(df.index, df['BB_upper'], linestyle='--', linewidth=0.5, color='blue')
    ax1.plot(df.index, df['BB_middle'], linewidth=0.5, color='blue')
    ax1.plot(df.index, df['BB_lower'], linestyle='--', linewidth=0.5, color='blue')

    for j in range(1, len(df)):
        color = 'green' if st['supertrend'].iloc[j] else 'red'
        ax1.axvspan(df.index[j-1], df.index[j], color=color, alpha=0.03)

    support_idx = argrelextrema(df['low'].values, np.less_equal, order=10)[0]
    resistance_idx = argrelextrema(df['high'].values, np.greater_equal, order=10)[0]
    support = df['low'].iloc[support_idx].tail(3)
    resistance = df['high'].iloc[resistance_idx].tail(3)
    x_pos = df.index[-1]

    offset_map = {'1m': pd.Timedelta(minutes=2), '5m': pd.Timedelta(minutes=10),
                  '15m': pd.Timedelta(minutes=20), '1h': pd.Timedelta(hours=1)}
    x_offset = offset_map.get(tf, pd.Timedelta(minutes=10))

    for s in support:
        ax1.axhline(s, color='green', linestyle='--', linewidth=0.5)
        ax1.text(x_pos + x_offset, s, f'{s:.2f}', va='center', ha='left', fontsize=7, color='green',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    for r in resistance:
        ax1.axhline(r, color='red', linestyle='--', linewidth=0.5)
        ax1.text(x_pos + x_offset, r, f'{r:.2f}', va='center', ha='left', fontsize=7, color='red',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    last_price = df['close'].iloc[-1]
    ax1.axhline(last_price, color='black', linestyle='--', linewidth=0.6)
    ax1.text(x_pos + x_offset, last_price, f'{last_price:.2f}', va='center', ha='left', fontsize=8, color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7))

    if not support.empty and last_price < support.min():
        ax1.annotate("⬇️ Breakdown", xy=(x_pos, last_price),
                     xytext=(x_pos, last_price * 1.01),
                     arrowprops=dict(arrowstyle="->", color='red'),
                     color='red', fontsize=9, ha='center')
    if not resistance.empty and last_price > resistance.max():
        ax1.annotate("⬆️ Breakout", xy=(x_pos, last_price),
                     xytext=(x_pos, last_price * 0.99),
                     arrowprops=dict(arrowstyle="->", color='green'),
                     color='green', fontsize=9, ha='center')

    ax1.set_title(f"{symbol} - {tf.upper()} Chart (Gate.io Futures)")
    ax1.legend(fontsize=6)
    ax1.grid(True)

    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', linewidth=0.5)
    ax2.axhline(30, color='green', linestyle='--', linewidth=0.5)

    ax2b = ax2.twinx()
    ax2b.plot(df.index, df['MACD'], label='MACD', color='black')
    ax2b.plot(df.index, df['MACD_signal'], label='Signal', color='orange', linestyle='--')
    ax2.set_title("RSI & MACD")
    ax2.legend(loc='upper left', fontsize=6)
    ax2b.legend(loc='upper right', fontsize=6)
    ax2.grid(True)

    colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    ax3.bar(df.index, df['volume'], color=colors, alpha=0.4, label='Volume')
    ax3.plot(df.index, df['Volume_MA20'], color='blue', linewidth=0.8, label='MA20')
    ax3.set_title("Volume")
    ax3.legend(fontsize=6)
    ax3.grid(True)

    fig.text(0.5, 0.5, "Gate.io Signal Bot", fontsize=40, color='gray', ha='center', va='center', alpha=0.1, rotation=30)
    plt.tight_layout(h_pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf
