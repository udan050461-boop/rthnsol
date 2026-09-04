import ccxt
import pandas as pd
import numpy as np
import ta
import time
import os
import requests # Tambahan pikeun Telegram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

load_dotenv()

# === KONFIGURASI ===
PAIRS = ['ETH/USDT', 'SOL/USDT']
TIMEFRAME = '1h' # m60
LIMIT_DATA = 1000 
CONFIDENCE_THRESHOLD = 0.90 # Filter 99%: Ngan trade lamun confidence > 90%
ATR_MULTIPLIER_SL = 0.5 # SL sa ipis-ipisna

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

exchange = ccxt.binance({'enableRateLimit': True})

def fetch_ohlcv(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT_DATA)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_features(df):
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=26)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'])
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['vol_sma'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def train_ai_model(df):
    features = ['ema_fast', 'ema_slow', 'rsi', 'macd', 'stoch', 'atr', 'bb_width', 
                'obv', 'vol_ratio', 'candle_body', 'candle_range', 'upper_wick', 'lower_wick']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🧠 AI Model trained. Validation Accuracy: {acc*100:.2f}%")
    return model, features

def send_telegram_message(message):
    """Fungsi pikeun ngirim pesen ka Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML" # Supados bisa bold, italic, jsb.
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Sinyal sukses dikirim ka Telegram!")
        else:
            print(f"❌ Gagal kirim ka Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Error koneksi Telegram: {e}")

def get_signal_and_sl(symbol, model, features):
    df = fetch_ohlcv(symbol)
    df = add_features(df)
    latest_data = df[features].iloc[[-1]]
    latest_price = df['close'].iloc[-1]
    latest_atr = df['atr'].iloc[-1]
    proba = model.predict_proba(latest_data)[0]
    prob_up = proba[1] 
    prob_down = proba[0]
    signal = None
    confidence = 0
    
    if prob_up > CONFIDENCE_THRESHOLD:
        signal = "LONG (BUY)"
        confidence = prob_up
        sl = latest_price - (latest_atr * ATR_MULTIPLIER_SL) 
    elif prob_down > CONFIDENCE_THRESHOLD:
        signal = "SHORT (SELL)"
        confidence = prob_down
        sl = latest_price + (latest_atr * ATR_MULTIPLIER_SL)
        
    return signal, confidence, latest_price, sl

def main():
    print("🚀 AI Quant Bot starting...")
    print(f"Pairs: {PAIRS} | Timeframe: {TIMEFRAME} | Confidence Filter: >{CONFIDENCE_THRESHOLD*100}%")
    print("-" * 50)
    
    models = {}
    feature_names = {}
    
    for pair in PAIRS:
        print(f"📊 Fetching data & training AI for {pair}...")
        df = fetch_ohlcv(pair)
        df = add_features(df)
        model, features = train_ai_model(df)
        models[pair] = model
        feature_names[pair] = features
        
    print("\n✅ AI Ready! Monitoring market & sending to Telegram...\n")
    
    while True:
        for pair in PAIRS:
            signal, conf, price, sl = get_signal_and_sl(pair, models[pair], feature_names[pair])
            
            if signal:
                # Format pesen pikeun Telegram
                emoji = "🟢" if "LONG" in signal else "🔴"
                msg = f"{emoji} <b>SINYAL AI 99% - {pair}</b>\n\n"
                msg += f"<b>Arah:</b> {signal}\n"
                msg += f"<b>Entry:</b> ${price:,.2f}\n"
                msg += f"<b>SL (Ipis):</b> ${sl:,.2f}\n"
                msg += f"<b>Confidence:</b> {conf*100:.2f}%\n\n"
                msg += f"<i>⚠️ Ulah lupa DYOR & manage risk!</i>"
                
                # Kirim ka Telegram
                send_telegram_message(msg)
                
                # Print ka terminal Codespaces ogé
                print(f"🚨 [SINYAL AI] {pair} -> {signal} @ {price} | SL: {sl:.2f} | Conf: {conf*100:.2f}%")
                print("-" * 50)
                
        # Tunggu 1 menit
        time.sleep(60)

if __name__ == "__main__":
    main()
