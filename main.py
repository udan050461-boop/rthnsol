import ccxt
import pandas as pd
import numpy as np
import ta
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from dotenv import load_dotenv

load_dotenv()

# === KONFIGURASI ===
PAIRS = ['ETH/USDT', 'SOL/USDT']
TIMEFRAME = '1h' # m60
LIMIT_DATA = 1000 # Data historical pikeun training AI
CONFIDENCE_THRESHOLD = 0.90 # Filter 99%: Ngan trade lamun confidence > 90%
ATR_MULTIPLIER_SL = 0.5 # SL sa ipis-ipisna (0.5 x ATR)

# Inisialisasi Exchange (Binance Public API, teu butuh API key pikeun data)
exchange = ccxt.binance({'enableRateLimit': True})

def fetch_ohlcv(symbol):
    """Nyokot data real-time ti exchange"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT_DATA)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_features(df):
    """Nambahkeun fitur teknikal pikeun AI (Leuwih ti 20 fitur)"""
    # Trend
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # Momentum
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    
    # Volatility (Penting pikeun SL ipis)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'])
    
    # Volume
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['vol_sma'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma']
    
    # Price Action
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    
    # Target Label: 1 lamun candle salajengna close leuwih luhur (AI diajar nebak arah)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    return df

def train_ai_model(df):
    """Nglatih uteuk AI (Random Forest)"""
    features = ['ema_fast', 'ema_slow', 'rsi', 'macd', 'stoch', 'atr', 'bb_width', 
                'obv', 'vol_ratio', 'candle_body', 'candle_range', 'upper_wick', 'lower_wick']
    
    X = df[features]
    y = df['target']
    
    # Train test split (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Cek akurasi model (Validation)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🧠 AI Model trained. Validation Accuracy: {acc*100:.2f}%")
    
    return model, features

def get_signal_and_sl(symbol, model, features):
    """Ngaeksekusi AI pikeun sinyal real-time"""
    df = fetch_ohlcv(symbol)
    df = add_features(df)
    
    # Data panganyarna pikeun prediksi
    latest_data = df[features].iloc[[-1]]
    latest_price = df['close'].iloc[-1]
    latest_atr = df['atr'].iloc[-1]
    
    # Prediksi probabilitas
    proba = model.predict_proba(latest_data)[0]
    
    # Kelas 0 = Turun, Kelas 1 = Naek
    prob_up = proba[1] 
    prob_down = proba[0]
    
    signal = None
    confidence = 0
    
    # FILTER 99%: Ngan asup lamun confidence leuwih ti threshold
    if prob_up > CONFIDENCE_THRESHOLD:
        signal = "LONG (BUY)"
        confidence = prob_up
        # SL SA IPIS-IPISNA: Entry - (0.5 * ATR)
        sl = latest_price - (latest_atr * ATR_MULTIPLIER_SL) 
    elif prob_down > CONFIDENCE_THRESHOLD:
        signal = "SHORT (SELL)"
        confidence = prob_down
        # SL SA IPIS-IPISNA: Entry + (0.5 * ATR)
        sl = latest_price + (latest_atr * ATR_MULTIPLIER_SL)
        
    return signal, confidence, latest_price, sl

def main():
    print("🚀 AI Quant Botstarting...")
    print(f"Pairs: {PAIRS} | Timeframe: {TIMEFRAME} | Confidence Filter: >{CONFIDENCE_THRESHOLD*100}%")
    print("-" * 50)
    
    models = {}
    feature_names = {}
    
    # Training AI pikeun tiap pair
    for pair in PAIRS:
        print(f"📊Fetching data & training AI for {pair}...")
        df = fetch_ohlcv(pair)
        df = add_features(df)
        model, features = train_ai_model(df)
        models[pair] = model
        feature_names[pair] = features
        
    print("\n✅ AI Ready! Monitoring market every 1 minute...\n")
    
    while True:
        for pair in PAIRS:
            signal, conf, price, sl = get_signal_and_sl(pair, models[pair], feature_names[pair])
            
            if signal:
                print(f"🚨 [SINYAL AI 99%] {pair}")
                print(f"   Arah     : {signal}")
                print(f"   Entry    : {price}")
                print(f"   SL (Tipis): {sl:.4f}")
                print(f"   Confidence: {conf*100:.2f}%")
                print("-" * 50)
                # Di dieu anjeun bisa nambihan logika pikeun kirim ka Telegram atawa eksekusi order real
                # exchange.create_order(...)
            else:
                # Teu aya sinyal (AI nyaring noise pasar)
                pass 
                
        # Tunggu 1 menit (sabab TF m60, tapi urang cek tiap menit pikeun update ATR/Price)
        time.sleep(60)

if __name__ == "__main__":
    main()
