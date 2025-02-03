import pandas as pd
import numpy as np
import requests
import time
import ta
from rich import print
from rich.table import Table
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ======== Configuration ========
SYMBOLS = ["BTCUSDT", "HBARUSDT", "SUIUSDT"]
TIMEFRAMES = ["15m", "30m", "1h"]
CHECK_INTERVAL = 600  # 10 minutes
RISK_REWARD_RATIO = 2.0
MAX_RISK_PERCENT = 1.0  # Maximum risk per trade (percentage of account balance)
ML_LOOKBACK = 100  # Bars for ML training
ACCOUNT_BALANCE = 10000  # Example account balance for position sizing

# ======== Data Fetching ========
def get_binance_data(symbol, interval="15m", limit=200):
    """Fetch OHLCV data from Binance"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[red]Error fetching {symbol} {interval} data: {e}[/red]")
        return None

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    return df.dropna()

# ======== Technical Indicators ========
def calculate_indicators(df):
    """Calculate technical indicators with error handling"""
    if df.empty:
        return df
    
    try:
        # Trend Indicators
        df["EMA_9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["EMA_21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
        df["EMA_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
        
        # Momentum Indicators
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Volatility Indicators
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], 14
        ).average_true_range()
        
        # Volume Indicators
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        
    except Exception as e:
        print(f"[red]Error calculating indicators: {e}[/red]")
        return df
    
    return df.dropna()

# ======== Machine Learning Integration ========
def prepare_ml_data(df):
    """Prepare dataframe for machine learning features"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(20).std()
    df['volume_change'] = df['volume'].pct_change()
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    return df.dropna()

def train_ml_model(df):
    """Train and return ML model with safety checks"""
    required_columns = ['returns', 'volatility', 'volume_change', 'RSI', 'MACD']
    missing = [col for col in required_columns if col not in df]
    if missing:
        raise ValueError(f"Missing columns for ML training: {missing}")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[required_columns])
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, scaler

# ======== Signal Detection ========
def detect_divergence(df):
    """Detect RSI divergences"""
    df = df.copy()
    df['price_high'] = df['high'].rolling(5).max()
    df['price_low'] = df['low'].rolling(5).min()
    df['rsi_high'] = df['RSI'].rolling(5).max()
    df['rsi_low'] = df['RSI'].rolling(5).min()
    
    bearish_div = (df['price_high'] > df['price_high'].shift(5)) & (df['rsi_high'] < df['rsi_high'].shift(5))
    bullish_div = (df['price_low'] < df['price_low'].shift(5)) & (df['rsi_low'] > df['rsi_low'].shift(5))
    
    return {
        "bearish_divergence": bearish_div.iloc[-1],
        "bullish_divergence": bullish_div.iloc[-1]
    }

def analyze_market(symbol_data, model_data):
    """Advanced signal detection"""
    signals = {
        "long": False, "short": False,
        "entry_price": None, "stop_loss": None,
        "take_profit": None, "confidence": 0,
        "ml_prediction": 0
    }
    
    # Access full DataFrames for each timeframe
    df_15m = symbol_data["15m"]
    df_30m = symbol_data["30m"]
    df_1h = symbol_data["1h"]
    
    # Get last row values
    tf15 = df_15m.iloc[-1]
    tf30 = df_30m.iloc[-1]
    tf1h = df_1h.iloc[-1]
    
    # Divergence detection
    divergence = detect_divergence(df_15m)
    
    # ML Prediction
    ml_df = prepare_ml_data(df_15m.copy())
    if len(ml_df) > ML_LOOKBACK and model_data:
        try:
            model, scaler = model_data["model"], model_data["scaler"]
            features = scaler.transform(ml_df[['returns', 'volatility', 'volume_change', 'RSI', 'MACD']].iloc[-1:])
            ml_pred = model.predict_proba(features)[0][1]
        except Exception as e:
            print(f"[red]ML prediction error: {e}[/red]")
            ml_pred = 0.5
    else:
        ml_pred = 0.5
    
    # Signal Conditions (using full DataFrames)
    primary_trend_up = tf1h["EMA_21"] > tf1h["EMA_50"]
    secondary_trend_up = tf30["EMA_21"] > tf30["EMA_50"]
    immediate_trend_up = tf15["EMA_9"] > tf15["EMA_21"]
    rsi_oversold = tf15["RSI"] < 35
    rsi_overbought = tf15["RSI"] > 65
    macd_bullish = tf15["MACD"] > tf15["MACD_Signal"]
    volume_spike = (tf15["volume"] > df_15m["volume"].rolling(20).mean().iloc[-1] * 1.5)
    
    long_conditions = [
        primary_trend_up,
        secondary_trend_up,
        immediate_trend_up,
        macd_bullish,
        rsi_oversold,
        volume_spike,
        divergence["bullish_divergence"],
        ml_pred > 0.6
    ]
    
    short_conditions = [
        not primary_trend_up,
        not secondary_trend_up,
        not immediate_trend_up,
        not macd_bullish,
        rsi_overbought,
        volume_spike,
        divergence["bearish_divergence"],
        ml_pred < 0.4
    ]
    
    long_score = sum(long_conditions)
    short_score = sum(short_conditions)
    
    if long_score >= 5:
        signals.update({
            "long": True,
            "entry_price": tf15["close"],
            "stop_loss": tf15["close"] - (tf15["ATR"] * 1.5),
            "take_profit": tf15["close"] + (tf15["ATR"] * RISK_REWARD_RATIO),
            "confidence": min(100, long_score * 15 + ml_pred * 20),
            "ml_prediction": ml_pred
        })
    elif short_score >= 5:
        signals.update({
            "short": True,
            "entry_price": tf15["close"],
            "stop_loss": tf15["close"] + (tf15["ATR"] * 1.5),
            "take_profit": tf15["close"] - (tf15["ATR"] * RISK_REWARD_RATIO),
            "confidence": min(100, short_score * 15 + (1 - ml_pred) * 20),
            "ml_prediction": ml_pred
        })
    
    return signals

# ======== Display & Risk Management ========
def dynamic_position_size(account_balance, entry_price, stop_loss, atr):
    """Volatility-adjusted position sizing"""
    risk_amount = account_balance * (MAX_RISK_PERCENT / 100)
    risk_per_share = abs(entry_price - stop_loss)
    atr_multiplier = max(0.5, min(2.0, 1.5 / (atr / entry_price)))
    return (risk_amount / risk_per_share) * atr_multiplier

def display_signals(all_signals):
    """Rich console output with additional trade details"""
    table = Table(title="\n🚀 Trading Signals", show_header=True, header_style="bold magenta")
    columns = [
        ("Metric", "cyan", 22),
        ("BTC", "green", 25),
        ("HBAR", "green", 25),
        ("SUI", "green", 25)
    ]
    
    for col in columns:
        table.add_column(col[0], style=col[1], width=col[2])
    
    # Prepare rows for each metric
    # Row 1: Current Price (from 15m timeframe)
    price_row = ["Price"]
    # Row 2: ML Prediction
    ml_row = ["ML Prediction"]
    # Row 3: Volatility (ATR)
    atr_row = ["Volatility (ATR)"]
    # Row 4: Signal (Long / Short / No Signal)
    signal_row = ["Signal"]
    # Row 5: Entry Price
    entry_row = ["Entry Price"]
    # Row 6: Stop Loss
    sl_row = ["Stop Loss"]
    # Row 7: Take Profit
    tp_row = ["Take Profit"]
    # Row 8: Estimated Position Size
    pos_size_row = ["Position Size"]

    for symbol in SYMBOLS:
        data = all_signals[symbol]
        tf15 = data["15m"].iloc[-1]
        sig = data["signal"]
        
        # Current Price, ATR, ML prediction
        price_row.append(f"{tf15['close']:.4f}")
        ml_row.append(f"{sig['ml_prediction']:.2%}")
        atr_row.append(f"{tf15['ATR']:.4f}")
        
        # Signal string formatting
        if sig["long"]:
            signal_row.append(f"[bold green]LONG ({sig['confidence']:.0f}%)[/bold green]")
        elif sig["short"]:
            signal_row.append(f"[bold red]SHORT ({sig['confidence']:.0f}%)[/bold red]")
        else:
            signal_row.append("[yellow]No Signal[/yellow]")
        
        # Suggested Entry, Stop Loss, and Take Profit values (if a signal exists)
        if sig["long"] or sig["short"]:
            entry_price = sig["entry_price"]
            stop_loss = sig["stop_loss"]
            take_profit = sig["take_profit"]
            entry_row.append(f"{entry_price:.4f}")
            sl_row.append(f"{stop_loss:.4f}")
            tp_row.append(f"{take_profit:.4f}")
            # Calculate position size using the dynamic_position_size function
            pos_size = dynamic_position_size(ACCOUNT_BALANCE, entry_price, stop_loss, tf15["ATR"])
            pos_size_row.append(f"{pos_size:.2f}")
        else:
            entry_row.append("N/A")
            sl_row.append("N/A")
            tp_row.append("N/A")
            pos_size_row.append("N/A")
    
    # Add rows to the table
    for row in [price_row, ml_row, atr_row, signal_row, entry_row, sl_row, tp_row, pos_size_row]:
        table.add_row(*row)
    
    print(table)

# ======== Main Execution ========
def main():
    print("[bold green]🚀 Starting Trading Scanner[/bold green]")
    
    # Initialize ML models with proper data processing
    ml_models = {}
    for symbol in SYMBOLS:
        try:
            df = get_binance_data(symbol, "15m", 500)
            if df is not None:
                df = calculate_indicators(df)
                prepared_df = prepare_ml_data(df)
                if not prepared_df.empty:
                    model, scaler = train_ml_model(prepared_df)
                    ml_models[symbol] = {"model": model, "scaler": scaler}
        except Exception as e:
            print(f"[red]Error initializing ML for {symbol}: {e}[/red]")
    
    while True:
        try:
            all_data = {}
            for symbol in SYMBOLS:
                symbol_data = {"symbol": symbol}
                for tf in TIMEFRAMES:
                    df = get_binance_data(symbol, tf)
                    if df is not None:
                        df = calculate_indicators(df)
                        symbol_data[tf] = df
                    time.sleep(0.1)
                all_data[symbol] = symbol_data
            
            processed_signals = {}
            for symbol in SYMBOLS:
                if all(tf in all_data[symbol] for tf in TIMEFRAMES):
                    processed_signals[symbol] = {
                        "signal": analyze_market(all_data[symbol], ml_models.get(symbol, {})),
                        "15m": all_data[symbol]["15m"],
                        "30m": all_data[symbol]["30m"],
                        "1h": all_data[symbol]["1h"]
                    }
            
            print(f"\n[white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/white]")
            display_signals(processed_signals)
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[bold red]⚠️  Scanner stopped by user[/bold red]")
            break
        except Exception as e:
            print(f"[red]Critical Error: {e}[/red]")
            time.sleep(60)

if __name__ == "__main__":
    main()
