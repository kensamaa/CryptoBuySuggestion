# Crypto Trading Scanner

A sophisticated cryptocurrency trading scanner that combines technical analysis, divergence detection, and machine learning to generate trading signals.

## Features

- **Multi-Timeframe Analysis**: Analyzes data across 15m, 30m, and 1h timeframes to confirm trends
- **Machine Learning Integration**: Uses ensemble machine learning models (RandomForest, GradientBoosting, LogisticRegression) with calibrated probability estimates
- **Technical Indicators**: Calculates various indicators including EMA, ADX, RSI, MACD, ATR, and OBV
- **Divergence Detection**: Identifies bullish and bearish divergences
- **Dynamic Position Sizing**: Calculates position sizes based on account balance, ATR, and risk parameters
- **Concurrent Data Fetching**: Uses multithreading for efficient data collection
- **Rich Terminal Output**: Displays signals with color-coded formatting

## Requirements

```
pandas
numpy
requests
ta
scikit-learn
rich
shap (optional, for explainability analysis)
```

## Configuration

The scanner can be configured by adjusting the following parameters:

```python
SYMBOLS = ["BTCUSDT", "HBARUSDT", "SUIUSDT"]  # Trading pairs to scan
TIMEFRAMES = ["15m", "30m", "1h"]             # Timeframes to analyze
CHECK_INTERVAL = 600                          # Scan interval (10 minutes)
RETRAIN_INTERVAL = 3600                       # ML model retraining interval (1 hour)
RISK_REWARD_RATIO = 2.0                       # Target profit vs stop loss ratio
MAX_RISK_PERCENT = 1.0                        # Maximum risk per trade
ML_LOOKBACK = 100                             # Bars for ML training
ACCOUNT_BALANCE = 1000                        # Account balance for position sizing
```

## Signal Generation Logic

The scanner generates signals based on multiple conditions:

1. Trend analysis across three timeframes
2. Technical indicator conditions (RSI, MACD, volume)
3. Divergence detection 
4. Machine learning predictions

A weighted scoring system determines signal confidence, with signals generated when the score reaches a minimum threshold.

## Usage

Run the script to start the scanner:

```bash
python trading_scanner.py
```

The scanner will:
1. Initialize ML models for each symbol
2. Fetch and analyze data at regular intervals
3. Display trading signals in a formatted table
4. Retrain ML models periodically to adapt to market conditions

## Sample Output

```
🚀 Trading Signals
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric               ┃ BTC                     ┃ HBAR                    ┃ SUI                     ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Price                │ 67450.5000              │ 0.0895                  │ 1.2340                  │
│ ML Prediction        │ 65.23%                  │ 43.12%                  │ 72.45%                  │
│ Avg ML Prediction    │ 63.47%                  │ 41.89%                  │ 70.31%                  │
│ Volatility (ATR)     │ 850.4200                │ 0.0025                  │ 0.0312                  │
│ Signal               │ LONG (85%)              │ No Signal               │ LONG (92%)              │
│ Entry Price          │ 67450.5000              │ N/A                     │ 1.2340                  │
│ Stop Loss            │ 66175.2700              │ N/A                     │ 1.1872                  │
│ Take Profit          │ 70001.0000              │ N/A                     │ 1.3276                  │
│ Position Size        │ 0.15                    │ N/A                     │ 784.45                  │
└──────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

## Risk Management

The scanner implements several risk management features:

1. Dynamic position sizing based on ATR and account balance
2. Maximum risk limit per trade
3. Automatic stop loss calculation
4. Take profit targets based on risk-reward ratio

## ML Model Training and Evaluation

Models are trained using recent price action and technical indicators. Key features include:
- Returns
- Volatility
- Volume changes
- Technical indicators (RSI, MACD)

The system periodically retrains models to adapt to changing market conditions.

## Disclaimer

This trading scanner is provided for educational and research purposes only. It is not financial advice. Always conduct your own research and risk assessment before trading.

## License

[MIT License](LICENSE)
