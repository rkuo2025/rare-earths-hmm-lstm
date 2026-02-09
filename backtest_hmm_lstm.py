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
    data = yf.download(ticker, start="2020-01-01", end=datetime.date.today().isoformat())
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data['Returns'] = data['Close'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    return data.dropna()

def backtest_hmm_lstm(ticker):
    print(f"--- Starting Backtest for {ticker} ---")
    df = get_data(ticker)
    if df is None: return
    
    # 1. HMM for Regimes
    hmm_features = df[['Returns', 'Range']].values
    scaler_hmm = StandardScaler()
    hmm_features_scaled = scaler_hmm.fit_transform(hmm_features)
    hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    hmm_model.fit(hmm_features_scaled)
    df['Regime'] = hmm_model.predict(hmm_features_scaled)
    
    # 2. LSTM Prep
    regime_dummies = pd.get_dummies(df['Regime'], prefix='Regime').astype(float)
    lstm_data = pd.concat([df[['Close']], regime_dummies], axis=1)
    scaler_lstm = MinMaxScaler()
    scaled_data = scaler_lstm.fit_transform(lstm_data)
    
    X, y = [], []
    lookback = 60
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 3. Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    
    # 4. Strategy & Metrics
    predictions = model.predict(X_test)
    dummy = np.zeros((len(predictions), lstm_data.shape[1]))
    dummy[:, 0] = predictions.flatten()
    inv_predictions = scaler_lstm.inverse_transform(dummy)[:, 0]
    
    # Actual prices for the test period
    actual_prices = df['Close'].values[split + lookback:]
    
    # Strategy: Buy if Predicted Price(t+1) > Actual Price(t)
    signals = []
    for i in range(len(inv_predictions) - 1):
        if inv_predictions[i+1] > actual_prices[i]:
            signals.append(1) # Long
        else:
            signals.append(0) # Cash
    signals.append(0) # Last day
    
    test_df = pd.DataFrame({
        'Actual': actual_prices,
        'Signal': signals
    }, index=df.index[split + lookback:])
    
    test_df['Daily_Return'] = test_df['Actual'].pct_change()
    test_df['Strategy_Return'] = test_df['Signal'].shift(1) * test_df['Daily_Return']
    test_df['Strategy_Return'] = test_df['Strategy_Return'].fillna(0)
    
    # Performance Stats
    test_df['Cum_Market'] = (1 + test_df['Daily_Return']).cumprod()
    test_df['Cum_Strategy'] = (1 + test_df['Strategy_Return']).cumprod()
    
    sharpe = (test_df['Strategy_Return'].mean() / test_df['Strategy_Return'].std()) * np.sqrt(252)
    total_return = (test_df['Cum_Strategy'].iloc[-1] - 1) * 100
    market_return = (test_df['Cum_Market'].iloc[-1] - 1) * 100
    
    print(f"\nResults for {ticker}:")
    print(f"Strategy Total Return: {total_return:.2f}%")
    print(f"Market Total Return: {market_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Plot Backtest
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['Cum_Market'], label='Market (Buy & Hold)')
    plt.plot(test_df['Cum_Strategy'], label='HMM-LSTM Strategy', color='green')
    plt.title(f'{ticker} Backtest Performance')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.savefig(f'{ticker.lower()}_backtest.png')
    
    return test_df

if __name__ == "__main__":
    for t in ["MP", "REMX"]:
        try:
            backtest_hmm_lstm(t)
        except Exception as e:
            print(f"Error backtesting {t}: {e}")
