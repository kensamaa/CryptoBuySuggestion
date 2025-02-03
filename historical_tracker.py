# ------------ historical_tracker.py ------------
import pandas as pd
from rich.console import Console
from rich.table import Table

class PerformanceTracker:
    def __init__(self):
        self.trade_history = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'direction', 
            'entry_price', 'exit_price', 'return_pct'
        ])
        self.equity_curve = []
        self.console = Console()

    def add_trade(self, entry_time, exit_time, direction, entry_price, exit_price):
        return_pct = ((exit_price - entry_price)/entry_price * 100) if direction == 'LONG' \
                    else ((entry_price - exit_price)/entry_price * 100)
        
        new_trade = pd.DataFrame([{
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': return_pct
        }])
        
        self.trade_history = pd.concat([self.trade_history, new_trade], ignore_index=True)
        self.equity_curve.append(self.equity_curve[-1]*(1 + return_pct/100) if self.equity_curve else 10000)

    def generate_report(self):
        table = Table(title="Performance Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Trades", str(len(self.trade_history)))
        table.add_row("Win Rate", f"{self.trade_history[self.trade_history['return_pct'] > 0].shape[0]/len(self.trade_history)*100:.1f}%")
        table.add_row("Max Drawdown", f"{self.calculate_max_drawdown():.1f}%")
        table.add_row("Profit Factor", f"{self.calculate_profit_factor():.2f}")
        
        self.console.print(table)
        self.plot_equity_curve()

    def calculate_max_drawdown(self):
        peak = self.equity_curve[0]
        max_dd = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value)/peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def calculate_profit_factor(self):
        gains = self.trade_history[self.trade_history['return_pct'] > 0]['return_pct'].sum()
        losses = abs(self.trade_history[self.trade_history['return_pct'] < 0]['return_pct'].sum())
        return gains/losses if losses != 0 else float('inf')

    def plot_equity_curve(self):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(self.equity_curve)
            plt.title("Equity Curve")
            plt.xlabel("Trades")
            plt.ylabel("Portfolio Value")
            plt.show()
        except ImportError:
            print("Matplotlib not installed, skipping chart")