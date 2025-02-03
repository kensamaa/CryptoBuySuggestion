import pandas as pd
import numpy as np
import requests
import time
import ta  # Technical Analysis library
from rich import print
from rich.table import Table
from datetime import datetime

# ======== Configuration ========
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
CHECK_INTERVAL = 600  # 10 minutes
RISK_REWARD_RATIO = 2.0
MAX_POSITION_SIZE = 0.1  # 10% of capital per trade

# ======== Enhanced Data Fetching ========
def get_binance_data(symbol=SYMBOL, interval=TIMEFRAME, limit=200):
    """Fetch OHLCV data with error handling and retries"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[red]Error fetching data: {e}[/red]")
        return None

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # Convert columns to numeric
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    return df.dropna()

# ======== Advanced Indicator Calculation ========
def calculate_indicators(df):
    """Calculate multiple technical indicators with confluence"""
    # Trend Indicators
    df["EMA_9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["EMA_21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    
    # Momentum Indicators
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["MACD"] = ta.trend.MACD(df["close"]).macd()
    df["MACD_Signal"] = ta.trend.MACD(df["close"]).macd_signal()
    
    # Volatility Indicators
    bollinger = ta.volatility.BollingerBands(df["close"], 20, 2)
    df["BB_Upper"] = bollinger.bollinger_hband()
    df["BB_Lower"] = bollinger.bollinger_lband()
    
    # Volume Indicators
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    
    # Support/Resistance
    df["Pivot"] = (df["high"] + df["low"] + df["close"]) / 3
    
    return df.dropna()

# ======== Enhanced Signal Detection ========
def analyze_market(df):
    """Advanced signal detection with multiple confirmations"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = {
        "long": False,
        "short": False,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "confidence": 0
    }
    
    # Trend Filter
    primary_trend_up = latest["EMA_21"] > latest["EMA_50"]
    secondary_trend_up = latest["EMA_9"] > latest["EMA_21"]
    
    # Long Signal Conditions
    long_conditions = [
        secondary_trend_up,
        latest["MACD"] > latest["MACD_Signal"],
        latest["RSI"] < 35,
        latest["close"] > latest["BB_Upper"],
        df["OBV"].iloc[-3:].is_monotonic_increasing,
        latest["volume"] > df["volume"].rolling(20).mean().iloc[-1]
    ]
    
    # Short Signal Conditions
    short_conditions = [
        not secondary_trend_up,
        latest["MACD"] < latest["MACD_Signal"],
        latest["RSI"] > 65,
        latest["close"] < latest["BB_Lower"],
        df["OBV"].iloc[-3:].is_monotonic_decreasing,
        latest["volume"] > df["volume"].rolling(20).mean().iloc[-1]
    ]
    
    # Signal Generation
    if sum(long_conditions) >= 4:
        signals["long"] = True
        signals["entry_price"] = latest["close"]
        signals["stop_loss"] = latest["BB_Lower"]
        signals["take_profit"] = latest["close"] + RISK_REWARD_RATIO * (latest["close"] - signals["stop_loss"])
        signals["confidence"] = min(100, sum(long_conditions)*20)
        
    elif sum(short_conditions) >= 4:
        signals["short"] = True
        signals["entry_price"] = latest["close"]
        signals["stop_loss"] = latest["BB_Upper"]
        signals["take_profit"] = latest["close"] - RISK_REWARD_RATIO * (signals["stop_loss"] - latest["close"])
        signals["confidence"] = min(100, sum(short_conditions)*20)
    
    return signals

# ======== Risk Management ========
def calculate_position_size(balance, entry_price, stop_loss):
    risk_amount = balance * MAX_POSITION_SIZE
    risk_per_unit = abs(entry_price - stop_loss)
    return risk_amount / risk_per_unit

# ======== Enhanced Visualization ========
def display_signals(signals, df):
    """Rich console output with formatted table"""
    table = Table(title="\n💰 Trading Signals", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    latest = df.iloc[-1]
    table.add_row("Timestamp", f"{df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    table.add_row("Price", f"{latest['close']:.2f}")
    table.add_row("Trend", "Bullish" if latest['EMA_9'] > latest['EMA_21'] else "Bearish")
    table.add_row("RSI", f"{latest['RSI']:.1f}")
    table.add_row("MACD Histogram", f"{latest['MACD'] - latest['MACD_Signal']:.2f}")
    
    if signals["long"] or signals["short"]:
        direction = "LONG" if signals["long"] else "SHORT"
        table.add_row("Signal", f"[bold]{direction}[/bold] (Confidence: {signals['confidence']}%)")
        table.add_row("Entry Price", f"{signals['entry_price']:.2f}")
        table.add_row("Stop Loss", f"{signals['stop_loss']:.2f}")
        table.add_row("Take Profit", f"{signals['take_profit']:.2f}")
    else:
        table.add_row("Signal", "[yellow]No Clear Opportunity[/yellow]")
    
    print(table)

# ======== Main Execution ========
def main():
    print("[bold green]🚀 Starting Enhanced Trading Signal Scanner[/bold green]")
    
    while True:
        try:
            df = get_binance_data()
            if df is None or df.empty:
                time.sleep(60)
                continue
                
            df = calculate_indicators(df)
            signals = analyze_market(df)
            
            print(f"\n[white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/white]")
            display_signals(signals, df)
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[bold red]⚠️  Scanner stopped by user[/bold red]")
            break
        except Exception as e:
            print(f"[red]Error: {e}[/red]")
            time.sleep(60)

if __name__ == "__main__":
    main()