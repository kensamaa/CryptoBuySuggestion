"""
Simple Logistic Regression Strategy
- Uses basic technical indicators as features
- Predicts probability of successful long/short
- Serves as interpretable baseline
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def create_features(df):
    """Feature Engineering with Technical Indicators"""
    features = pd.DataFrame(index=df.index)
    
    # Price Features
    features['returns'] = df['close'].pct_change()
    features['volatility'] = df['close'].rolling(20).std()
    
    # Technical Indicators
    features['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    features['macd'] = ta.trend.MACD(df['close']).macd_diff()
    features['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    # Target: Next period return > threshold (1% in 4h)
    features['target'] = (df['close'].pct_change(4).shift(-4) > 0.01).astype(int)
    
    return features.dropna()

def train_model(features):
    """Train-Test Split and Model Training"""
    X = features.drop('target', axis=1)
    y = features['target']
    
    # Temporal Split (avoid lookahead)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model Training
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluation
    print("Train Score:", model.score(X_train, y_train))
    print(classification_report(y_test, model.predict(X_test)))
    
    return model, scaler

def live_prediction(model, scaler, new_data):
    """Real-time Prediction"""
    features = create_features(new_data).iloc[-1:]
    features_scaled = scaler.transform(features.drop('target', axis=1))
    proba = model.predict_proba(features_scaled)[0]
    return {'long_prob': proba[1], 'features': features}


"""
Advanced Gradient Boosting Implementation
- Handles non-linear relationships
- Automatic feature importance
- Robust to outliers
"""

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def train_boosted_model(features):
    X = features.drop('target', axis=1)
    y = features['target']
    
    # Time-Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8
    }
    
    for train_idx, test_idx in tscv.split(X):
        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        test_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
        
        model = lgb.train(params,
                         train_data,
                         valid_sets=[test_data],
                         num_boost_round=1000,
                         early_stopping_rounds=50)
        
    # Feature Importance
    lgb.plot_importance(model, importance_type='gain')
    return model

"""
LSTM Implementation for Temporal Patterns
- Captures sequential dependencies
- Requires careful feature engineering
- Needs GPU acceleration for training
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(data, window_size=60):
    """Create time-series sequences"""
    X, y = [], []
    for i in range(len(data)-window_size-1):
        X.append(data[i:(i+window_size)])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

# Usage:
# scaled_data = ... # Normalized features
# X, y = create_sequences(scaled_data)
# model = build_lstm_model((X.shape[1], X.shape[2]))
# model.fit(X, y, epochs=50, batch_size=64, validation_split=0.2)

"""
Transformer Implementation for Market Attention
- Captures long-range dependencies
- Self-attention mechanism
- Requires large datasets
"""

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = TransformerBlock(64, 4, 128)(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)