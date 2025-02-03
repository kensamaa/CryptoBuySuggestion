import pandas as pd
import numpy as np
import requests
import ta
from datetime import datetime
from rich import print
from rich.table import Table
from rich.console import Console

# Initialize Rich console for logging
console = Console()

# ======== Configuration ========
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"
INITIAL_CAPITAL = 100.0      # Starting capital in dollars
RISK_REWARD_RATIO = 2.0      # For take profit calculation
MAX_RISK_PERCENT = 1.0       # Percent of current capital risked per trade
ML_LOOKBACK = 100            # Bars to skip initially (warm-up period)
ROLLING_VOLUME_PERIOD = 20   # Period for rolling volume average

# ======== Data Fetching ========
def get_binance_data(symbol, interval="15m", limit=1000):
    """Fetch historical OHLCV data from Binance"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        console.log(f"[red]Error fetching {symbol} {interval} data: {e}[/red]")
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
    """Calculate indicators needed for the strategy."""
    if df.empty:
        return df
    try:
        # Trend indicators
        df["EMA_9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["EMA_21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
        df["EMA_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        # Momentum indicators
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        # Volatility indicator
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], 14
        ).average_true_range()
        # Rolling volume mean for volume spike detection
        df["Volume_Mean"] = df["volume"].rolling(ROLLING_VOLUME_PERIOD).mean()
    except Exception as e:
        console.log(f"[red]Error calculating indicators: {e}[/red]")
    return df.dropna()

# ======== Signal Detection & Backtest Simulation ========
def backtest_strategy(df):
    """
    Loop through the historical data and simulate a trade (either long or short)
    whenever the strategy conditions are met.
    """
    capital = INITIAL_CAPITAL
    trades = []
    win_count = 0
    loss_count = 0

    # Loop through the data, starting after an initial warm-up period.
    for i in range(ML_LOOKBACK, len(df) - 1):
        row = df.iloc[i]
        atr = row["ATR"]
        if atr <= 0:
            continue  # Skip if ATR is non-positive

        # Calculate conditions for long signals:
        long_primary_trend_up = row["EMA_21"] > row["EMA_50"]
        long_immediate_trend_up = row["EMA_9"] > row["EMA_21"]
        long_rsi_oversold = row["RSI"] < 35
        long_macd_bullish = row["MACD"] > row["MACD_Signal"]
        volume_spike = row["volume"] > row["Volume_Mean"] * 1.5

        long_conditions = [
            long_primary_trend_up,
            long_immediate_trend_up,
            long_rsi_oversold,
            long_macd_bullish,
            volume_spike
        ]

        # Calculate conditions for short signals (inverse of long conditions):
        short_primary_trend_down = row["EMA_21"] < row["EMA_50"]
        short_immediate_trend_down = row["EMA_9"] < row["EMA_21"]
        short_rsi_overbought = row["RSI"] > 65
        short_macd_bearish = row["MACD"] < row["MACD_Signal"]

        short_conditions = [
            short_primary_trend_down,
            short_immediate_trend_down,
            short_rsi_overbought,
            short_macd_bearish,
            volume_spike
        ]

        # Determine if we have a long or short signal (only one at a time)
        trade_type = None
        if sum(long_conditions) >= 4 and sum(short_conditions) < 4:
            trade_type = "LONG"
        elif sum(short_conditions) >= 4 and sum(long_conditions) < 4:
            trade_type = "SHORT"
        else:
            continue  # Skip if signals are ambiguous or none is strong enough

        entry_price = row["close"]

        # Define trade parameters based on trade type
        if trade_type == "LONG":
            stop_loss = entry_price - atr * 1.5
            take_profit = entry_price + atr * RISK_REWARD_RATIO
            # Risk per unit is the difference between entry and stop (long direction)
            risk_per_unit = entry_price - stop_loss  
        else:  # SHORT
            stop_loss = entry_price + atr * 1.5
            take_profit = entry_price - atr * RISK_REWARD_RATIO
            # For short trades, risk per unit is the difference between stop and entry
            risk_per_unit = stop_loss - entry_price  

        if risk_per_unit == 0:
            continue

        # Determine position size based on risk (risking MAX_RISK_PERCENT of current capital)
        risk_amount = capital * (MAX_RISK_PERCENT / 100)
        position_size = risk_amount / risk_per_unit

        # Simulate the trade outcome by scanning forward:
        trade_outcome = None
        exit_price = None
        exit_index = None
        for j in range(i+1, len(df)):
            trade_row = df.iloc[j]
            if trade_type == "LONG":
                # For a long trade, if the low hits or goes below stop loss, it's a loss.
                if trade_row["low"] <= stop_loss:
                    trade_outcome = "loss"
                    exit_price = stop_loss
                    exit_index = j
                    break
                # If the high reaches or exceeds the take profit, it's a win.
                elif trade_row["high"] >= take_profit:
                    trade_outcome = "win"
                    exit_price = take_profit
                    exit_index = j
                    break
            else:  # SHORT trade
                # For a short trade, if the high hits or goes above stop loss, it's a loss.
                if trade_row["high"] >= stop_loss:
                    trade_outcome = "loss"
                    exit_price = stop_loss
                    exit_index = j
                    break
                # If the low reaches or falls below the take profit, it's a win.
                elif trade_row["low"] <= take_profit:
                    trade_outcome = "win"
                    exit_price = take_profit
                    exit_index = j
                    break

        if trade_outcome is None:
            continue

        # Calculate profit in dollars for this trade:
        if trade_type == "LONG":
            profit = (exit_price - entry_price) * position_size
        else:  # SHORT
            profit = (entry_price - exit_price) * position_size

        if profit >= 0:
            win_count += 1
        else:
            loss_count += 1

        capital += profit

        trades.append({
            "trade_type": trade_type,
            "entry_index": i,
            "exit_index": exit_index,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit": profit,
            "capital_after_trade": capital,
            "outcome": trade_outcome
        })

    return trades, win_count, loss_count, capital

# ======== Display Results Using Rich ========
def display_results(trades, wins, losses, final_capital):
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (losses / total_trades * 100) if total_trades > 0 else 0
    profit_percentage = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    summary_table = Table(title="Backtest Summary", style="bold magenta")
    summary_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Total Trades", str(total_trades))
    summary_table.add_row("Wins", f"{wins} ({win_rate:.2f}%)")
    summary_table.add_row("Losses", f"{losses} ({loss_rate:.2f}%)")
    summary_table.add_row("Final Capital", f"${final_capital:.2f}")
    summary_table.add_row("Profit/Loss", f"{profit_percentage:.2f}%")

    console.print(summary_table)

    # Detailed trades table
    trades_table = Table(title="Trades Detail", style="bold blue")
    trades_table.add_column("Type", justify="center")
    trades_table.add_column("Entry Index", justify="right")
    trades_table.add_column("Exit Index", justify="right")
    trades_table.add_column("Entry Price", justify="right")
    trades_table.add_column("Exit Price", justify="right")
    trades_table.add_column("Profit", justify="right")
    trades_table.add_column("Outcome", justify="center")

    for trade in trades:
        trades_table.add_row(
            trade["trade_type"],
            str(trade["entry_index"]),
            str(trade["exit_index"]),
            f"{trade['entry_price']:.2f}",
            f"{trade['exit_price']:.2f}",
            f"{trade['profit']:.2f}",
            trade["outcome"].upper()
        )
    
    console.print(trades_table)

# ======== Main Execution ========
def main():
    console.print("[bold green]Starting Backtest...[/bold green]")
    df = get_binance_data(SYMBOL, TIMEFRAME, limit=1000)
    if df is None or df.empty:
        console.log("[red]No data available for backtesting.[/red]")
        return

    df = calculate_indicators(df)
    
    trades, wins, losses, final_capital = backtest_strategy(df)
    
    display_results(trades, wins, losses, final_capital)

if __name__ == "__main__":
    main()
