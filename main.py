import pandas as pd
import numpy as np
import requests
import time
import ta
import warnings
import logging
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

# Suppress RuntimeWarnings from numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======== Logging Configuration ========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======== Global Configuration ========
SYMBOLS = ["BTCUSDT", "HBARUSDT", "SEIUSDT"]
TIMEFRAMES = ["15m", "30m", "1h"]
CHECK_INTERVAL = 600         # seconds (10 minutes)
RETRAIN_INTERVAL = 3600      # seconds (1 hour)
RISK_REWARD_RATIO = 2.0
MAX_RISK_PERCENT = 1.0       # percent risk per trade
ML_LOOKBACK = 100            # bars for ML training
ACCOUNT_BALANCE = 1000       # account balance example

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

@app.route("/")
def dashboard():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Crypto Trading Dashboard</title>
      <!-- Bootstrap CSS -->
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style>
        body { background-color: #f8f9fa; }
        h1 { margin-top: 20px; }
        .table thead th { background-color: #343a40; color: white; }
        .card { margin-bottom: 20px; }
      </style>
      <!-- Chart.js -->
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <script>
      async function fetchData() {
          const response = await fetch('/api/data');
          const data = await response.json();
          let tableBody = '';
          for (let symbol in data) {
              let row = data[symbol];
              tableBody += `<tr>
                  <td>${symbol}</td>
                  <td>${row.price !== null ? row.price : 'N/A'}</td>
                  <td>${row.ml_prediction !== null ? (row.ml_prediction * 100).toFixed(2) + '%' : 'N/A'}</td>
                  <td>${row.avg_ml_prediction !== null ? (row.avg_ml_prediction * 100).toFixed(2) + '%' : 'N/A'}</td>
                  <td>${row.atr !== null ? row.atr : 'N/A'}</td>
                  <td>${row.signal !== null ? row.signal : 'N/A'}</td>
                  <td>${row.entry_price !== null ? row.entry_price : 'N/A'}</td>
                  <td>${row.stop_loss !== null ? row.stop_loss : 'N/A'}</td>
                  <td>${row.take_profit !== null ? row.take_profit : 'N/A'}</td>
                  <td>${row.position_size !== null ? row.position_size : 'N/A'}</td>
              </tr>`;
          }
          document.getElementById('data-table-body').innerHTML = tableBody;
      }
      
      async function fetchEquity() {
          const response = await fetch('/api/equity');
          const equityData = await response.json();
          const ctx = document.getElementById('equityChart').getContext('2d');
          if (window.myChart) {
              window.myChart.data.labels = equityData.map((_, index) => index);
              window.myChart.data.datasets[0].data = equityData;
              window.myChart.update();
          } else {
              window.myChart = new Chart(ctx, {
                  type: 'line',
                  data: {
                      labels: equityData.map((_, index) => index),
                      datasets: [{
                          label: 'Equity Curve',
                          data: equityData,
                          borderColor: 'rgba(75, 192, 192, 1)',
                          backgroundColor: 'rgba(75, 192, 192, 0.2)',
                          fill: true,
                          tension: 0.1
                      }]
                  },
                  options: {
                      scales: {
                          x: { title: { display: true, text: 'Trades' } },
                          y: { title: { display: true, text: 'Portfolio Value' } }
                      }
                  }
              });
          }
      }
      
      function refreshDashboard() {
          fetchData();
          fetchEquity();
      }
      
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
                </tr>
              </thead>
              <tbody id="data-table-body">
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route("/api/data")
def api_data():
    return jsonify(dashboard_data)

@app.route("/api/equity")
def api_equity():
    # Return the equity curve from the performance tracker
    return jsonify(perf_tracker.equity_curve)

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

# ======== Signal Detection ========
def detect_divergence(df: pd.DataFrame) -> Dict[str, bool]:
    df = df.copy()
    df['price_high'] = df['high'].rolling(5).max()
    df['price_low'] = df['low'].rolling(5).min()
    df['rsi_high'] = df['RSI'].rolling(5).max()
    df['rsi_low'] = df['RSI'].rolling(5).min()
    bearish_div = (df['price_high'] > df['price_high'].shift(5)) & (df['rsi_high'] < df['rsi_high'].shift(5))
    bullish_div = (df['price_low'] < df['price_low'].shift(5)) & (df['rsi_low'] > df['rsi_low'].shift(5))
    if not df.empty:
        return {
            "bearish_divergence": bool(bearish_div.iloc[-1]),
            "bullish_divergence": bool(bullish_div.iloc[-1])
        }
    return {"bearish_divergence": False, "bullish_divergence": False}

def analyze_market(symbol_data: Dict[str, Any], model_data: Dict[str, Any]) -> Dict[str, Any]:
    signals = {
        "long": False, "short": False,
        "entry_price": None, "stop_loss": None,
        "take_profit": None, "confidence": 0,
        "ml_prediction": 0
    }
    
    df_15m = symbol_data["15m"]
    df_30m = symbol_data["30m"]
    df_1h = symbol_data["1h"]
    
    tf15 = df_15m.iloc[-1]
    tf30 = df_30m.iloc[-1]
    tf1h = df_1h.iloc[-1]
    
    divergence = detect_divergence(df_15m)
    
    ml_df = prepare_ml_data(df_15m.copy())
    if len(ml_df) > ML_LOOKBACK and model_data:
        try:
            model, scaler = model_data["model"], model_data["scaler"]
            features = scaler.transform(ml_df[['returns', 'volatility', 'volume_change', 'RSI', 'MACD']].iloc[-1:])
            ml_pred = model.predict_proba(features)[0][1]
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            ml_pred = 0.5
    else:
        ml_pred = 0.5
    
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

def dynamic_position_size(account_balance: float, entry_price: float, stop_loss: float, atr: float) -> float:
    risk_amount = account_balance * (MAX_RISK_PERCENT / 100)
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        return 0.0
    atr_multiplier = max(0.5, min(2.0, 1.5 / (atr / entry_price)))
    return (risk_amount / risk_per_share) * atr_multiplier

def display_signals(all_signals: Dict[str, Any],
                    prediction_history: Dict[str, deque]) -> None:
    table = Table(title="\n🚀 Trading Signals", show_header=True, header_style="bold magenta")
    columns = [
        ("Metric", "cyan", 22),
        ("BTC", "green", 25),
        ("HBAR", "green", 25),
        ("SEI", "green", 25)
    ]
    for col in columns:
        table.add_column(col[0], style=col[1], width=col[2])
    
    price_row = ["Price"]
    ml_row = ["ML Prediction"]
    avg_ml_row = ["Avg ML Prediction"]
    atr_row = ["Volatility (ATR)"]
    signal_row = ["Signal"]
    entry_row = ["Entry Price"]
    sl_row = ["Stop Loss"]
    tp_row = ["Take Profit"]
    pos_size_row = ["Position Size"]

    for symbol in SYMBOLS:
        data = all_signals.get(symbol, {})
        tf15 = data.get("15m")
        sig = data.get("signal", {})
        if tf15 is None or not sig:
            price_row.append("N/A")
            ml_row.append("N/A")
            avg_ml_row.append("N/A")
            atr_row.append("N/A")
            signal_row.append("N/A")
            entry_row.append("N/A")
            sl_row.append("N/A")
            tp_row.append("N/A")
            pos_size_row.append("N/A")
            continue
        
        tf15_last = tf15.iloc[-1]
        price_row.append(f"{tf15_last['close']:.4f}")
        ml_value = sig.get("ml_prediction", 0)
        ml_row.append(f"{ml_value:.2%}")
        history = prediction_history.get(symbol, [])
        avg_ml = np.mean(history) if history else 0
        avg_ml_row.append(f"{avg_ml:.2%}")
        atr_row.append(f"{tf15_last['ATR']:.4f}")
        
        if sig.get("long"):
            signal_row.append(f"[bold green]LONG ({sig.get('confidence', 0):.0f}%)[/bold green]")
        elif sig.get("short"):
            signal_row.append(f"[bold red]SHORT ({sig.get('confidence', 0):.0f}%)[/bold red]")
        else:
            signal_row.append("[yellow]No Signal[/yellow]")
        
        if sig.get("long") or sig.get("short"):
            entry_price = sig.get("entry_price", 0)
            stop_loss = sig.get("stop_loss", 0)
            take_profit = sig.get("take_profit", 0)
            entry_row.append(f"{entry_price:.4f}")
            sl_row.append(f"{stop_loss:.4f}")
            tp_row.append(f"{take_profit:.4f}")
            pos_size = dynamic_position_size(ACCOUNT_BALANCE, entry_price, stop_loss, tf15_last["ATR"])
            pos_size_row.append(f"{pos_size:.2f}")
        else:
            entry_row.append("N/A")
            sl_row.append("N/A")
            tp_row.append("N/A")
            pos_size_row.append("N/A")
    
    for row in [price_row, ml_row, avg_ml_row, atr_row, signal_row, entry_row, sl_row, tp_row, pos_size_row]:
        table.add_row(*row)
    
    print(table)

def update_dashboard(processed_signals: Dict[str, Any], prediction_history: Dict[str, deque]) -> None:
    global dashboard_data
    new_data = {}
    for symbol in SYMBOLS:
        data = processed_signals.get(symbol, {})
        tf15 = data.get("15m")
        sig = data.get("signal", {})
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
                "position_size": None
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
            "position_size": round(pos_size, 2) if pos_size else None
        }
    dashboard_data = new_data

def fetch_symbol_data(symbol: str, tf: str, session: requests.Session) -> Tuple[str, str, Optional[pd.DataFrame]]:
    df = get_binance_data(symbol, tf, session=session)
    if df is not None:
        df = calculate_indicators(df)
    return symbol, tf, df

# ======== Main Execution ========
def main() -> None:
    logger.info("🚀 Starting Trading Scanner")
    session = requests.Session()  # Reuse HTTP connections
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
            display_signals(processed_signals, prediction_history)
            update_dashboard(processed_signals, prediction_history)
            
            # Optionally, update performance tracker (e.g., after a trade is completed)
            # perf_tracker.add_trade(entry_time, exit_time, direction, entry_price, exit_price)
            
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
