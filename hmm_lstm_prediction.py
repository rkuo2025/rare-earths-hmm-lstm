import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import datetime

def get_data(ticker):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end=datetime.date.today().isoformat())
    if data.empty:
        return None
    
    # Flatten multi-index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    # Use returns and volatility as features for HMM
    data['Returns'] = data['Close'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    return data.dropna()

def train_hmm_lstm(ticker):
    df = get_data(ticker)
    if df is None: return
    
    # --- Part 1: Hidden Markov Model (HMM) ---
    # We use HMM to identify market regimes (e.g., Bull, Bear, Sideways)
    hmm_features = df[['Returns', 'Range']].values
    scaler_hmm = StandardScaler()
    hmm_features_scaled = scaler_hmm.fit_transform(hmm_features)
    
    # 3 regimes
    hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    hmm_model.fit(hmm_features_scaled)
    regimes = hmm_model.predict(hmm_features_scaled)
    
    # Add regimes to dataframe
    df['Regime'] = regimes
    
    # --- Part 2: LSTM ---
    # Features for LSTM: Close price + HMM Regimes
    # One-hot encode regimes
    regime_dummies = pd.get_dummies(df['Regime'], prefix='Regime').astype(float)
    lstm_data = pd.concat([df[['Close']], regime_dummies], axis=1)
    
    scaler_lstm = MinMaxScaler()
    scaled_data = scaler_lstm.fit_transform(lstm_data)
    
    # Create sequences
    X, y = [], []
    lookback = 60
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0]) # Predict next Close
        
    X, y = np.array(X), np.array(y)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f"Training LSTM for {ticker}...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # Prediction
    predictions = model.predict(X_test)
    
    # Denormalize
    # Create a dummy array to inverse scale correctly
    dummy = np.zeros((len(predictions), lstm_data.shape[1]))
    dummy[:, 0] = predictions.flatten()
    inv_predictions = scaler_lstm.inverse_transform(dummy)[:, 0]
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-len(y_test):], df['Close'].values[-len(y_test):], label='Actual')
    plt.plot(df.index[-len(inv_predictions):], inv_predictions, label='Predicted')
    plt.title(f'{ticker} Hybrid HMM-LSTM Prediction')
    plt.legend()
    plt.savefig(f'{ticker.lower()}_hmm_lstm.png')
    print(f"Prediction for {ticker} saved.")

if __name__ == "__main__":
    # USA Rare Earths is private, so we use REMX (Rare Earth/Strategic Metals ETF) as a proxy
    tickers = ["MP", "REMX"]
    for t in tickers:
        try:
            train_hmm_lstm(t)
        except Exception as e:
            print(f"Error for {t}: {e}")
