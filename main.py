import pandas as pd
import numpy as np
import requests
import time
import ta
import warnings
import logging
import random
from rich import print
from rich.table import Table
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from typing import Optional, Dict, Any, Tuple, List
import threading

# Flask & JSON for Dashboard
from flask import Flask, jsonify, render_template_string

# Plotly for interactive coin charts
import plotly.graph_objects as go

# Suppress RuntimeWarnings from numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======== Logging Configuration ========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======== Global Configuration ========
SYMBOLS = ["BTCUSDT", "HBARUSDT", "SUIUSDT"]
TIMEFRAMES = ["15m", "30m", "1h"]
CHECK_INTERVAL = 600         # seconds (10 minutes)
RETRAIN_INTERVAL = 3600      # seconds (1 hour)
RISK_REWARD_RATIO = 2.0
MAX_RISK_PERCENT = 1.0       # percent risk per trade
ML_LOOKBACK = 100            # bars for ML training
ACCOUNT_BALANCE = 1000       # example account balance

# Global storage for dashboard data
dashboard_data: Dict[str, Any] = {}

# ======== Performance Tracker ==========
class PerformanceTracker:
    """
    Tracks trade performance, maintains trade history and an equity curve,
    and provides both console reporting and programmatic access to key metrics.
    """
    def __init__(self, initial_balance: float = 10000):
        self.trade_history = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'direction', 
            'entry_price', 'exit_price', 'return_pct'
        ])
        self.equity_curve = [initial_balance]
        from rich.console import Console
        self.console = Console()

    def add_trade(self, entry_time, exit_time, direction, entry_price, exit_price):
        if direction.upper() == 'LONG':
            return_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            return_pct = ((entry_price - exit_price) / entry_price) * 100

        new_trade = pd.DataFrame([{
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': return_pct
        }])
        
        self.trade_history = pd.concat([self.trade_history, new_trade], ignore_index=True)
        new_equity = self.equity_curve[-1] * (1 + return_pct / 100)
        self.equity_curve.append(new_equity)
        return new_trade

    def calculate_max_drawdown(self):
        peak = self.equity_curve[0]
        max_dd = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def calculate_profit_factor(self):
        gains = self.trade_history[self.trade_history['return_pct'] > 0]['return_pct'].sum()
        losses = abs(self.trade_history[self.trade_history['return_pct'] < 0]['return_pct'].sum())
        return gains / losses if losses != 0 else float('inf')

    def generate_report(self):
        table = Table(title="Performance Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        total_trades = len(self.trade_history)
        win_count = self.trade_history[self.trade_history['return_pct'] > 0].shape[0]
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        table.add_row("Total Trades", str(total_trades))
        table.add_row("Win Rate", f"{win_rate:.1f}%")
        table.add_row("Max Drawdown", f"{self.calculate_max_drawdown():.1f}%")
        table.add_row("Profit Factor", f"{self.calculate_profit_factor():.2f}")
        self.console.print(table)
        self.plot_equity_curve()

    def plot_equity_curve(self):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(self.equity_curve, marker='o')
            plt.title("Equity Curve")
            plt.xlabel("Trades")
            plt.ylabel("Portfolio Value")
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib not installed, skipping chart.")

    def get_report_data(self):
        total_trades = len(self.trade_history)
        win_count = self.trade_history[self.trade_history['return_pct'] > 0].shape[0]
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "max_drawdown": round(self.calculate_max_drawdown(), 1),
            "profit_factor": round(self.calculate_profit_factor(), 2),
            "latest_equity": round(self.equity_curve[-1], 2)
        }

# Global Performance Tracker instance
perf_tracker = PerformanceTracker(initial_balance=10000)

# ======== Flask Dashboard Setup ==========
app = Flask(__name__)

def generate_dashboard_html():
    # Generate nav tabs for coin charts
    nav_tabs = ""
    tab_content = ""
    for s in SYMBOLS:
        nav_tabs += f'<li class="nav-item"><a class="nav-link" id="{s}-tab" data-toggle="tab" href="#{s}" role="tab">{s}</a></li>'
        tab_content += f'<div class="tab-pane fade" id="{s}" role="tabpanel"><div id="chart-{s}" style="width:100%;height:400px;"></div></div>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Crypto Trading Dashboard</title>
      <!-- Bootstrap CSS -->
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style>
        body {{ background-color: #f8f9fa; }}
        h1 {{ margin-top: 20px; }}
        .table thead th {{ background-color: #343a40; color: white; }}
        .card {{ margin-bottom: 20px; }}
      </style>
      <!-- Chart.js for equity curve -->
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <!-- Plotly JS for coin charts -->
      <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
      <script>
      async function fetchData() {{
          const response = await fetch('/api/data');
          const data = await response.json();
          let tableBody = '';
          for (let symbol in data) {{
              let row = data[symbol];
              tableBody += `<tr>
                  <td>${{symbol}}</td>
                  <td>${{row.price !== null ? row.price : 'N/A'}}</td>
                  <td>${{row.ml_prediction !== null ? (row.ml_prediction * 100).toFixed(2) + '%' : 'N/A'}}</td>
                  <td>${{row.avg_ml_prediction !== null ? (row.avg_ml_prediction * 100).toFixed(2) + '%' : 'N/A'}}</td>
                  <td>${{row.atr !== null ? row.atr : 'N/A'}}</td>
                  <td>${{row.signal !== null ? row.signal : 'N/A'}}</td>
                  <td>${{row.entry_price !== null ? row.entry_price : 'N/A'}}</td>
                  <td>${{row.stop_loss !== null ? row.stop_loss : 'N/A'}}</td>
                  <td>${{row.take_profit !== null ? row.take_profit : 'N/A'}}</td>
                  <td>${{row.position_size !== null ? row.position_size : 'N/A'}}</td>
                  <td>P:${{row.pivot !== null ? row.pivot : 'N/A'}} S:${{row.support !== null ? row.support : 'N/A'}} R:${{row.resistance !== null ? row.resistance : 'N/A'}}</td>
              </tr>`;
          }}
          document.getElementById('data-table-body').innerHTML = tableBody;
      }}
      
      async function fetchEquity() {{
          const response = await fetch('/api/equity');
          const equityData = await response.json();
          const ctx = document.getElementById('equityChart').getContext('2d');
          if (window.myChart) {{
              window.myChart.data.labels = equityData.map((_, index) => index);
              window.myChart.data.datasets[0].data = equityData;
              window.myChart.update();
          }} else {{
              window.myChart = new Chart(ctx, {{
                  type: 'line',
                  data: {{
                      labels: equityData.map((_, index) => index),
                      datasets: [{{
                          label: 'Equity Curve',
                          data: equityData,
                          borderColor: 'rgba(75, 192, 192, 1)',
                          backgroundColor: 'rgba(75, 192, 192, 0.2)',
                          fill: true,
                          tension: 0.1
                      }}]
                  }},
                  options: {{
                      scales: {{
                          x: {{ title: {{ display: true, text: 'Trades' }} }},
                          y: {{ title: {{ display: true, text: 'Portfolio Value' }} }}
                      }}
                  }}
              }});
          }}
      }}
      
      async function fetchCoinChart(symbol) {{
          const response = await fetch(`/api/chart/${{symbol}}`);
          const chartData = await response.json();
          Plotly.newPlot('chart-' + symbol, chartData.data, chartData.layout);
      }}
      
      function refreshDashboard() {{
          fetchData();
          fetchEquity();
          const symbols = {SYMBOLS};
          symbols.forEach(symbol => {{
              fetchCoinChart(symbol);
          }});
      }}
      
      setInterval(refreshDashboard, 10000);
      window.onload = refreshDashboard;
      </script>
    </head>
    <body>
      <div class="container">
        <h1 class="text-center text-primary">Crypto Trading Dashboard</h1>
        
        <!-- Equity Curve Chart -->
        <div class="card">
          <div class="card-body">
            <canvas id="equityChart" style="max-height:400px;"></canvas>
          </div>
        </div>
        
        <!-- Trading Signals Table -->
        <div class="card">
          <div class="card-body">
            <table class="table table-striped table-bordered">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Price</th>
                  <th>ML Prediction</th>
                  <th>Avg ML Prediction</th>
                  <th>ATR</th>
                  <th>Signal</th>
                  <th>Entry Price</th>
                  <th>Stop Loss</th>
                  <th>Take Profit</th>
                  <th>Position Size</th>
                  <th>P/S/R</th>
                </tr>
              </thead>
              <tbody id="data-table-body">
              </tbody>
            </table>
          </div>
        </div>
        
        <!-- Coin Charts -->
        <div class="card">
          <div class="card-body">
            <h3>Coin Charts</h3>
            <ul class="nav nav-tabs" id="chartTabs" role="tablist">
              {nav_tabs}
            </ul>
            <div class="tab-content" id="chartTabsContent">
              {tab_content}
            </div>
          </div>
        </div>
        
        <!-- Explanation Section -->
        <div class="card">
          <div class="card-body">
            <h3>Bot Workflow & Signal Criteria</h3>
            <p>
              <strong>Signal Criteria:</strong> The bot uses a weighted consensus from 15m, 30m, and 1h timeframes.
              Longer timeframes (1h) are given higher weight for trend confirmation while shorter timeframes (15m) are used for entry timing.
              Conditions include EMA crossovers, RSI thresholds, MACD signals, divergence with volume confirmation, and volume spikes.
            </p>
            <p>
              <strong>ML Prediction & Volatility:</strong> An ML model (trained on 15m data) predicts the probability of price increase.
              The average ML prediction is factored into the final signal. Volatility is measured via ATR, which influences position sizing, stop loss, and take profit.
            </p>
            <p>
              <strong>Bot Workflow:</strong> The bot fetches multi-timeframe data, computes technical indicators (including pivot levels, support/resistance, divergence),
              and then combines these with ML predictions using a weighted consensus to generate trading signals.
              Simulated trades are executed based on these signals, and the portfolio (equity curve) is updated accordingly.
            </p>
          </div>
        </div>
      </div>
      
      <!-- Bootstrap JS and dependencies -->
      <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    return html

@app.route("/")
def dashboard():
    return render_template_string(generate_dashboard_html())

@app.route("/api/data")
def api_data():
    return jsonify(dashboard_data)

@app.route("/api/equity")
def api_equity():
    # Return the equity curve from the performance tracker
    return jsonify(perf_tracker.equity_curve)

@app.route("/api/chart/<symbol>")
def api_chart(symbol):
    # Fetch 15m data for the symbol and generate a Plotly candlestick chart with support/resistance and a marker for the current signal
    session = requests.Session()
    df = get_binance_data(symbol, "15m", limit=200, session=session)
    if df is None or df.empty:
        return jsonify({})
    df = calculate_indicators(df)
    pivot_levels = get_pivot_levels(df)
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name="Price")])
    # Add support/resistance lines
    fig.add_hline(y=pivot_levels["pivot"], line_dash="dash", annotation_text="Pivot", annotation_position="bottom right")
    fig.add_hline(y=pivot_levels["support"], line_dash="dot", annotation_text="Support", annotation_position="bottom right")
    fig.add_hline(y=pivot_levels["resistance"], line_dash="dot", annotation_text="Resistance", annotation_position="top right")
    # Add buy/sell marker if signal exists
    sig = dashboard_data.get(symbol, {})
    if sig and sig.get("signal") in ["LONG", "SHORT"]:
        marker_color = "green" if sig.get("signal") == "LONG" else "red"
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[sig.get("entry_price", df['close'].iloc[-1])],
                                 mode="markers", marker=dict(color=marker_color, size=12),
                                 name=f"{sig.get('signal')} Entry"))
    return fig.to_json()

def run_dashboard():
    app.run(host="0.0.0.0", port=5000)

# ======== Data Fetching & Indicator Calculations ========
def get_binance_data(symbol: str, interval: str = "15m", limit: int = 200,
                     session: Optional[requests.Session] = None) -> Optional[pd.DataFrame]:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        sess = session or requests
        response = sess.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching {symbol} {interval} data: {e}")
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        df["EMA_9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["EMA_21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
        df["EMA_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def get_pivot_levels(df: pd.DataFrame) -> Dict[str, float]:
    # Compute pivot point, support and resistance from the last row
    last_row = df.iloc[-1]
    pivot = (last_row['high'] + last_row['low'] + last_row['close']) / 3
    resistance = (2 * pivot) - last_row['low']
    support = (2 * pivot) - last_row['high']
    return {"pivot": pivot, "support": support, "resistance": resistance}

# ======== Machine Learning Integration ========
def train_ml_model(df: pd.DataFrame) -> Tuple[CalibratedClassifierCV, StandardScaler]:
    required_columns = ['returns', 'volatility', 'volume_change', 'RSI', 'MACD']
    missing = [col for col in required_columns if col not in df]
    if missing:
        raise ValueError(f"Missing columns for ML training: {missing}")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[required_columns])
    y = df['target']
    
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf3 = LogisticRegression(max_iter=1000, solver='lbfgs')
    
    ensemble = VotingClassifier(estimators=[
        ('rf', clf1), ('gb', clf2), ('lr', clf3)
    ], voting='soft')
    
    calibrated = CalibratedClassifierCV(ensemble, cv=5)
    calibrated.fit(X, y)
    
    feature_names = required_columns
    importances: List[np.ndarray] = []
    for estimator in getattr(calibrated.base_estimator, 'estimators_', []):
        if hasattr(estimator, 'feature_importances_'):
            importances.append(estimator.feature_importances_)
    if importances:
        avg_importances = np.mean(importances, axis=0)
        imp_dict = dict(zip(feature_names, avg_importances))
        logger.info(f"Feature Importances: {imp_dict}")
    else:
        logger.info("No feature importances available from ensemble members.")
    
    try:
        import shap
        for name, estimator in calibrated.base_estimator.named_estimators_.items():
            if hasattr(estimator, 'predict_proba'):
                explainer = shap.TreeExplainer(estimator)
                sample = df[required_columns].iloc[-1:]
                features = scaler.transform(sample)
                shap_vals = explainer.shap_values(features)
                shap_contrib = dict(zip(feature_names, shap_vals[1][0]))
                logger.info(f"SHAP contributions for {name}: {shap_contrib}")
                break
    except Exception as e:
        logger.warning(f"SHAP analysis skipped: {e}")
    
    return calibrated, scaler

def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(20).std()
    df['volume_change'] = df['volume'].pct_change()
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# ======== Signal Detection & Weighted Consensus ========
def detect_divergence(df: pd.DataFrame) -> Dict[str, bool]:
    df = df.copy()
    df['price_high'] = df['high'].rolling(5).max()
    df['price_low'] = df['low'].rolling(5).min()
    volume_mean = df['volume'].rolling(5).mean().iloc[-1]
    # Basic divergence using current vs. shifted RSI and price highs/lows with volume confirmation
    bearish_div = (df['price_high'] > df['price_high'].shift(5)) & (df['RSI'] < df['RSI'].shift(5))
    bullish_div = (df['price_low'] < df['price_low'].shift(5)) & (df['RSI'] > df['RSI'].shift(5))
    bearish_confirm = bearish_div.iloc[-1] and (df['volume'].iloc[-1] > volume_mean * 1.5)
    bullish_confirm = bullish_div.iloc[-1] and (df['volume'].iloc[-1] > volume_mean * 1.5)
    return {
        "bearish_divergence": bool(bearish_confirm),
        "bullish_divergence": bool(bullish_confirm)
    }

def analyze_market(symbol_data: Dict[str, Any], model_data: Dict[str, Any]) -> Dict[str, Any]:
    # Weighted consensus: assign weights to each timeframe
    timeframe_weights = {"15m": 0.5, "30m": 0.75, "1h": 1.0}
    long_score = 0.0
    short_score = 0.0
    ml_preds = []
    for tf in TIMEFRAMES:
        if tf not in symbol_data:
            continue
        df = symbol_data[tf]
        latest = df.iloc[-1]
        weight = timeframe_weights.get(tf, 1)
        # Long conditions
        trend_long = latest["EMA_21"] > latest["EMA_50"]
        immediate_long = (latest["EMA_9"] > latest["EMA_21"]) and (latest["RSI"] < 35) and (latest["MACD"] > latest["MACD_Signal"])
        volume_spike = latest["volume"] > df["volume"].rolling(20).mean().iloc[-1] * 1.5
        divergence = detect_divergence(df)
        bullish_div = divergence["bullish_divergence"]
        # Short conditions
        trend_short = latest["EMA_21"] < latest["EMA_50"]
        immediate_short = (latest["EMA_9"] < latest["EMA_21"]) and (latest["RSI"] > 65) and (latest["MACD"] < latest["MACD_Signal"])
        bearish_div = divergence["bearish_divergence"]
        
        if trend_long:
            long_score += weight * 1
        if immediate_long:
            long_score += weight * 1
        if bullish_div:
            long_score += weight * 0.5
        if volume_spike:
            long_score += weight * 0.5

        if trend_short:
            short_score += weight * 1
        if immediate_short:
            short_score += weight * 1
        if bearish_div:
            short_score += weight * 0.5
        if volume_spike:
            short_score += weight * 0.5

        # Use ML prediction from 15m if available
        if tf == "15m" and model_data:
            try:
                ml_df = prepare_ml_data(df.copy())
                if len(ml_df) > ML_LOOKBACK:
                    model, scaler = model_data["model"], model_data["scaler"]
                    features = scaler.transform(ml_df[['returns', 'volatility', 'volume_change', 'RSI', 'MACD']].iloc[-1:])
                    ml_pred = model.predict_proba(features)[0][1]
                    ml_preds.append(ml_pred)
            except Exception as e:
                logger.error(f"ML prediction error for {tf}: {e}")
    
    avg_ml_pred = np.mean(ml_preds) if ml_preds else 0.5
    long_score += avg_ml_pred * 2
    short_score += (1 - avg_ml_pred) * 2

    threshold = 2.5
    if long_score >= threshold and long_score > short_score:
        signal = {
            "long": True,
            "entry_price": symbol_data["15m"].iloc[-1]["close"],
            "stop_loss": symbol_data["15m"].iloc[-1]["close"] - (symbol_data["15m"].iloc[-1]["ATR"] * 1.5),
            "take_profit": symbol_data["15m"].iloc[-1]["close"] + (symbol_data["15m"].iloc[-1]["ATR"] * RISK_REWARD_RATIO),
            "confidence": min(100, long_score * 15),
            "ml_prediction": avg_ml_pred
        }
    elif short_score >= threshold and short_score > long_score:
        signal = {
            "short": True,
            "entry_price": symbol_data["15m"].iloc[-1]["close"],
            "stop_loss": symbol_data["15m"].iloc[-1]["close"] + (symbol_data["15m"].iloc[-1]["ATR"] * 1.5),
            "take_profit": symbol_data["15m"].iloc[-1]["close"] - (symbol_data["15m"].iloc[-1]["ATR"] * RISK_REWARD_RATIO),
            "confidence": min(100, short_score * 15),
            "ml_prediction": avg_ml_pred
        }
    else:
        signal = {"long": False, "short": False, "ml_prediction": avg_ml_pred}
    
    return signal

def dynamic_position_size(account_balance: float, entry_price: float, stop_loss: float, atr: float) -> float:
    risk_amount = account_balance * (MAX_RISK_PERCENT / 100)
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        return 0.0
    atr_multiplier = max(0.5, min(2.0, 1.5 / (atr / entry_price)))
    return (risk_amount / risk_per_share) * atr_multiplier

def update_dashboard(processed_signals: Dict[str, Any], prediction_history: Dict[str, deque]) -> None:
    global dashboard_data
    new_data = {}
    for symbol in SYMBOLS:
        data = processed_signals.get(symbol, {})
        tf15 = data.get("15m")
        sig = data.get("signal", {})
        support_resistance = {"pivot": None, "support": None, "resistance": None}
        if "1h" in data and not data["1h"].empty:
            support_resistance = get_pivot_levels(data["1h"])
        if tf15 is None or not sig:
            new_data[symbol] = {
                "price": None,
                "ml_prediction": None,
                "avg_ml_prediction": None,
                "atr": None,
                "signal": None,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "position_size": None,
                "pivot": support_resistance["pivot"],
                "support": support_resistance["support"],
                "resistance": support_resistance["resistance"]
            }
            continue
        tf15_last = tf15.iloc[-1]
        price = tf15_last["close"]
        ml_prediction = sig.get("ml_prediction", 0)
        history = prediction_history.get(symbol, [])
        avg_ml_prediction = np.mean(history) if history else 0
        atr = tf15_last["ATR"]
        signal_str = "LONG" if sig.get("long") else ("SHORT" if sig.get("short") else "No Signal")
        entry_price = sig.get("entry_price")
        stop_loss = sig.get("stop_loss")
        take_profit = sig.get("take_profit")
        pos_size = dynamic_position_size(ACCOUNT_BALANCE, entry_price, stop_loss, atr) if (sig.get("long") or sig.get("short")) else None
        new_data[symbol] = {
            "price": round(price, 4),
            "ml_prediction": round(ml_prediction, 4),
            "avg_ml_prediction": round(avg_ml_prediction, 4),
            "atr": round(atr, 4),
            "signal": signal_str,
            "entry_price": round(entry_price, 4) if entry_price else None,
            "stop_loss": round(stop_loss, 4) if stop_loss else None,
            "take_profit": round(take_profit, 4) if take_profit else None,
            "position_size": round(pos_size, 2) if pos_size else None,
            "pivot": round(support_resistance["pivot"], 4) if support_resistance["pivot"] else None,
            "support": round(support_resistance["support"], 4) if support_resistance["support"] else None,
            "resistance": round(support_resistance["resistance"], 4) if support_resistance["resistance"] else None,
        }
    dashboard_data = new_data

def fetch_symbol_data(symbol: str, tf: str, session: requests.Session) -> Tuple[str, str, Optional[pd.DataFrame]]:
    df = get_binance_data(symbol, tf, session=session)
    if df is not None:
        df = calculate_indicators(df)
    return symbol, tf, df

# ======== Console Display Function ========
def display_signals_console(data: Dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title="Trading Signals (Terminal)")
    table.add_column("Symbol", style="bold cyan")
    table.add_column("Price", justify="right")
    table.add_column("Signal", justify="right")
    table.add_column("ML Prediction", justify="right")
    table.add_column("ATR", justify="right")
    table.add_column("Entry Price", justify="right")
    table.add_column("Stop Loss", justify="right")
    table.add_column("Take Profit", justify="right")
    table.add_column("Position Size", justify="right")
    table.add_column("Pivot", justify="right")
    table.add_column("Support", justify="right")
    table.add_column("Resistance", justify="right")
    
    for symbol, info in data.items():
        ml_pred = info.get("ml_prediction")
        ml_str = f"{ml_pred:.2%}" if ml_pred is not None else "N/A"
        table.add_row(
            symbol,
            str(info.get("price", "N/A")),
            str(info.get("signal", "N/A")),
            ml_str,
            str(info.get("atr", "N/A")),
            str(info.get("entry_price", "N/A")),
            str(info.get("stop_loss", "N/A")),
            str(info.get("take_profit", "N/A")),
            str(info.get("position_size", "N/A")),
            str(info.get("pivot", "N/A")),
            str(info.get("support", "N/A")),
            str(info.get("resistance", "N/A"))
        )
    console.print(table)

# ======== Main Execution ========
def main() -> None:
    logger.info("🚀 Starting Trading Scanner")
    session = requests.Session()
    ml_models: Dict[str, Dict[str, Any]] = {}
    prediction_history: Dict[str, deque] = {symbol: deque(maxlen=10) for symbol in SYMBOLS}

    # Initialize ML models per symbol using 15m data.
    for symbol in SYMBOLS:
        df = get_binance_data(symbol, "15m", limit=500, session=session)
        if df is not None:
            df = calculate_indicators(df)
            prepared_df = prepare_ml_data(df)
            if not prepared_df.empty:
                try:
                    model, scaler = train_ml_model(prepared_df)
                    ml_models[symbol] = {"model": model, "scaler": scaler}
                    logger.info(f"ML model initialized for {symbol}")
                except Exception as e:
                    logger.error(f"Error initializing ML for {symbol}: {e}")
    
    last_retrain = time.time()

    # Start the dashboard server in a separate thread.
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    logger.info("Dashboard running on http://0.0.0.0:5000")
    
    while True:
        try:
            all_data: Dict[str, Dict[str, Any]] = {symbol: {} for symbol in SYMBOLS}
            with ThreadPoolExecutor(max_workers=len(SYMBOLS) * len(TIMEFRAMES)) as executor:
                futures = []
                for symbol in SYMBOLS:
                    for tf in TIMEFRAMES:
                        futures.append(executor.submit(fetch_symbol_data, symbol, tf, session))
                for future in as_completed(futures):
                    symbol, tf, df = future.result()
                    if df is not None:
                        all_data[symbol][tf] = df
            
            processed_signals: Dict[str, Any] = {}
            for symbol in SYMBOLS:
                symbol_data = all_data.get(symbol, {})
                if all(tf in symbol_data for tf in TIMEFRAMES):
                    signal = analyze_market(symbol_data, ml_models.get(symbol, {}))
                    processed_signals[symbol] = {
                        "signal": signal,
                        "15m": symbol_data["15m"],
                        "30m": symbol_data["30m"],
                        "1h": symbol_data["1h"]
                    }
                    prediction_history[symbol].append(signal.get("ml_prediction", 0))
                else:
                    logger.warning(f"Incomplete data for {symbol}. Skipping signal analysis.")
            
            print(f"\n[white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/white]")
            # Display signals on terminal using Rich
            update_dashboard(processed_signals, prediction_history)
            display_signals_console(dashboard_data)
            
            # --- Trade Simulation ---
            for symbol in SYMBOLS:
                sig = processed_signals.get(symbol, {}).get("signal", {})
                if sig.get("long") or sig.get("short"):
                    entry_price = sig.get("entry_price")
                    stop_loss = sig.get("stop_loss")
                    take_profit = sig.get("take_profit")
                    if not entry_price or not stop_loss or not take_profit:
                        continue
                    if sig.get("long"):
                        win_prob = sig.get("ml_prediction", 0.5)
                        direction = "LONG"
                    else:
                        win_prob = 1 - sig.get("ml_prediction", 0.5)
                        direction = "SHORT"
                    r = random.random()
                    if r < win_prob:
                        exit_price = take_profit
                        outcome = "win"
                    else:
                        exit_price = stop_loss
                        outcome = "loss"
                    trade_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    perf_tracker.add_trade(entry_time=trade_time, exit_time=trade_time, direction=direction, entry_price=entry_price, exit_price=exit_price)
                    logger.info(f"Simulated {direction} trade for {symbol}: {outcome} | Entry: {entry_price}, Exit: {exit_price}")
            
            # Dashboard will automatically reflect updated equity curve via /api/equity
            
            if time.time() - last_retrain >= RETRAIN_INTERVAL:
                logger.info("Retraining ML models...")
                for symbol in SYMBOLS:
                    df = get_binance_data(symbol, "15m", limit=500, session=session)
                    if df is not None:
                        df = calculate_indicators(df)
                        prepared_df = prepare_ml_data(df)
                        if not prepared_df.empty:
                            try:
                                model, scaler = train_ml_model(prepared_df)
                                ml_models[symbol] = {"model": model, "scaler": scaler}
                                logger.info(f"ML model retrained for {symbol}")
                            except Exception as e:
                                logger.error(f"Error retraining ML for {symbol}: {e}")
                                continue
                last_retrain = time.time()
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n[bold red]⚠️  Scanner stopped by user[/bold red]")
            break
        except Exception as e:
            logger.error(f"Critical Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
