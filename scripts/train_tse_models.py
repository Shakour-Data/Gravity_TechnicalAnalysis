"""
Train ML models for TSE data.

This script trains pattern classification models using TSE market data.
"""

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from gravity_tech.database.database_manager import DatabaseManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_tse_data():
    """Load TSE data from database."""
    db = DatabaseManager()
    conn = db.get_connection()

    query = """
    SELECT symbol, timestamp, open, high, low, close, volume
    FROM price_data
    WHERE symbol LIKE 'IRO%'
    ORDER BY symbol, timestamp
    LIMIT 10000
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def extract_features(df):
    """Extract technical analysis features."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol', 'timestamp'])

    # Basic price features
    df['returns'] = df.groupby('symbol')['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df.groupby('symbol')['close'].shift(1))

    # Moving averages
    df['sma_20'] = df.groupby('symbol')['close'].rolling(20).mean()
    df['sma_50'] = df.groupby('symbol')['close'].rolling(50).mean()

    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['rsi'] = df.groupby('symbol')['close'].apply(calculate_rsi)

    # MACD
    df['ema_12'] = df.groupby('symbol')['close'].ewm(span=12).mean()
    df['ema_26'] = df.groupby('symbol')['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('symbol')['macd'].ewm(span=9).mean()

    # Target: simple pattern classification (bullish/bearish/neutral)
    df['target'] = np.where(df['returns'] > 0.02, 1, np.where(df['returns'] < -0.02, -1, 0))

    # Drop NaN
    df = df.dropna()

    features = ['returns', 'log_returns', 'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal']
    X = df[features]
    y = df['target']

    return X, y

def train_models(X, y):
    """Train multiple ML models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)
    models['lightgbm'] = lgb_model

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    models['xgboost'] = xgb_model

    # Evaluate
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name.upper()} Results:")
        print(classification_report(y_test, y_pred))

    return models

def save_models(models, path='ml_models/tse/'):
    """Save trained models."""
    os.makedirs(path, exist_ok=True)

    for name, model in models.items():
        with open(f'{path}{name}_tse.pkl', 'wb') as f:
            pickle.dump(model, f)

    print(f"Models saved to {path}")

if __name__ == "__main__":
    print("Loading TSE data...")
    df = load_tse_data()
    print(f"Loaded {len(df)} records")

    print("Extracting features...")
    X, y = extract_features(df)
    print(f"Features shape: {X.shape}, Target distribution: {y.value_counts()}")

    print("Training models...")
    models = train_models(X, y)

    print("Saving models...")
    save_models(models)

    print("Training complete!")