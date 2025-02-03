# ------------ multi_timeframe_analyzer.py ------------
class MultiTimeframeAnalyzer:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.timeframes = {
            '1h': 3600,
            '15m': 900,
            '5m': 300
        }
        
    def get_confluence(self):
        confluence = {'trend': 'neutral', 'momentum': 'neutral'}
        
        # Fetch data for multiple timeframes
        data = {}
        for tf in self.timeframes:
            data[tf] = get_binance_data(symbol=self.symbol, interval=tf)
        
        # Analyze trend alignment
        trends = []
        for tf in ['1h', '15m', '5m']:
            trends.append(self._get_trend_direction(data[tf]))
        
        confluence['trend'] = 'bullish' if trends.count('bullish') > 1 else 'bearish' if trends.count('bearish') > 1 else 'neutral'
        
        # Analyze momentum convergence
        rsi_values = [self._get_rsi_strength(data[tf]) for tf in ['1h', '15m']]
        confluence['momentum'] = 'bullish' if sum(rsi_values) > 0 else 'bearish'
        
        return confluence
    
    def _get_trend_direction(self, df):
        ema21 = df['EMA_21'].iloc[-1]
        ema50 = df['EMA_50'].iloc[-1]
        return 'bullish' if ema21 > ema50 else 'bearish'
    
    def _get_rsi_strength(self, df):
        rsi = df['RSI'].iloc[-1]
        if rsi > 70: return -1
        if rsi < 30: return 1
        return 0