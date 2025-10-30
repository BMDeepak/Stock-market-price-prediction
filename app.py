from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import traceback

# Set random seed for reproducibility
np.random.seed(42)

app = Flask(__name__)
# Allow all origins for local development (adjust for production)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load pre-trained models
gru_model = joblib.load('gru_model.pkl')
gru_scaler = joblib.load('gru_scaler.pkl')

# Data Collection
def fetch_stock_data(ticker, start_date='2020-01-01', end_date=datetime.today()):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"No data fetched for ticker {ticker}. Check ticker or network connection.")
    if not all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        raise ValueError("Fetched data missing required columns.")
    return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Feature Engineering
def engineer_features(data):
    print("Engineering features")
    df = data.copy()
    close_array = df['Close'].to_numpy().ravel().astype(np.float64)
    df['SMA_20'] = talib.SMA(close_array, timeperiod=20)
    df['RSI'] = talib.RSI(close_array, timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
    df = df.dropna()
    return df

# Prepare data for Prophet
def prepare_prophet_data(data, with_features=True):
    print("Preparing data for Prophet")
    df_prophet = data[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']
    if with_features:
        regressors = ['SMA_20', 'RSI', 'MACD']
        for reg in regressors:
            df_prophet[reg] = data[reg].values
    return df_prophet

# Train Prophet Model and Predict (new instance for each fit)
def train_prophet_model(df_prophet, forecast_days=5, with_features=True):
    print(f"Training new Prophet model for {forecast_days} days")
    # Create a new Prophet instance
    model = Prophet(daily_seasonality=True)
    if with_features:
        regressors = ['SMA_20', 'RSI', 'MACD']
        for reg in regressors:
            model.add_regressor(reg)
    
    test_size = 30
    train_data = df_prophet[:-test_size]
    test_data = df_prophet[-test_size:]
    
    model.fit(train_data)
    
    future_test = test_data.drop(columns=['y'])
    forecast_test = model.predict(future_test)
    test_preds = forecast_test['yhat'].values
    
    # Generate future dates, filter out weekends and holidays
    start_date = datetime(2025, 6, 23, 16, 8)  # Next trading day after June 22, 2025
    all_dates = pd.date_range(start=start_date, periods=forecast_days * 2, freq='D')  # Overshoot to ensure enough days
    trading_dates = [d for d in all_dates if d.weekday() < 5 and d not in get_us_holidays()]  # Monday=0, Sunday=6
    future_dates = trading_dates[:forecast_days]
    
    future_df = pd.DataFrame({'ds': future_dates})
    if with_features:
        for reg in regressors:
            future_df[reg] = df_prophet[reg].iloc[-1]
    future_forecast = model.predict(future_df)
    future_preds = future_forecast['yhat'].values
    
    return test_preds, future_preds, test_data['y'].values, test_data['ds'], future_dates

# Prepare data for GRU
def prepare_gru_data(data, with_features=True, seq_length=60, test_size=30):
    print("Preparing data for GRU")
    if with_features:
        features = ['Close', 'SMA_20', 'RSI', 'MACD']
    else:
        features = ['Close']
    df = data[features]
    scaled_data = gru_scaler.transform(df)  # Use pre-trained scaler
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Insufficient data for GRU sequence creation.")
    test_start = max(0, len(data) - test_size - seq_length)
    X_train, X_test = X[:test_start], X[test_start:]
    y_train, y_test = y[:test_start], y[test_start:]
    return X_train, X_test, y_train, y_test, scaled_data, gru_scaler

# Predict on test set for GRU (using pre-trained model)
def predict_gru_test(model, X_test, scaler, with_features=True):
    print("Predicting test set")
    y_pred_scaled = model.predict(X_test, verbose=0)
    if with_features:
        num_features = 4
    else:
        num_features = 1
    y_pred_full = np.zeros((len(y_pred_scaled), num_features))
    y_pred_full[:, 0] = y_pred_scaled[:, 0]
    y_pred = scaler.inverse_transform(y_pred_full)[:, 0]
    return y_pred

# Inverse transform y for actual values
def inverse_transform_y(y, scaler, with_features=True):
    print("Inverse transforming y")
    if with_features:
        num_features = 4
    else:
        num_features = 1
    y_full = np.zeros((len(y), num_features))
    y_full[:, 0] = y
    y_actual = scaler.inverse_transform(y_full)[:, 0]
    return y_actual

# Predict future for GRU with dynamic feature updates and increased perturbation
def predict_gru_future(model, last_sequence, scaler, data, forecast_days=20, with_features=True):
    print(f"Predicting future for {forecast_days} days")
    predictions = []
    current_sequence = last_sequence.copy()
    if current_sequence.shape[0] < 60:
        raise ValueError(f"Insufficient sequence length: {current_sequence.shape[0]}. Expected at least 60.")
    
    last_price = scaler.inverse_transform([last_sequence[-1]])[0, 0]
    max_growth = last_price * 1.5
    min_growth = last_price * 0.5
    
    predicted_prices = scaler.inverse_transform(last_sequence)[-60:, 0].tolist()
    
    # Generate trading days for predictions
    start_date = datetime(2025, 6, 23, 16, 8)
    all_dates = pd.date_range(start=start_date, periods=forecast_days * 2, freq='D')
    trading_dates = [d for d in all_dates if d.weekday() < 5 and d not in get_us_holidays()]
    forecast_dates = trading_dates[:forecast_days]
    
    for i in range(forecast_days):
        pred_scaled = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0, 0]
        if np.isnan(pred_scaled):
            print(f"NaN prediction at step {i}. Using last valid value.")
            pred_scaled = 0.0 if len(predictions) == 0 else predictions[-1]
        if with_features:
            num_features = 4
        else:
            num_features = 1
        pred_full = np.zeros((1, num_features))
        pred_full[0, 0] = pred_scaled
        pred = scaler.inverse_transform(pred_full)[0, 0]
        
        perturbation = np.random.normal(0, 0.02 * last_price)
        pred += perturbation
        pred = max(min(pred, max_growth), min_growth)
        predictions.append(pred)
        
        predicted_prices.append(pred)
        if len(predicted_prices) > 60:
            predicted_prices = predicted_prices[-60:]
        
        close_array = np.array(predicted_prices).astype(np.float64)
        sma_20 = talib.SMA(close_array, timeperiod=20)[-1] if len(close_array) >= 20 else close_array[-1]
        rsi = talib.RSI(close_array, timeperiod=14)[-1] if len(close_array) >= 14 else 50.0
        macd, macd_signal, _ = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = macd[-1] if len(macd) > 0 else 0.0
        
        new_features = np.array([[pred, sma_20, rsi, macd]])
        new_scaled = scaler.transform(new_features)[0]
        
        new_row = new_scaled if with_features else np.array([pred_scaled])
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    return predictions, forecast_dates

# Evaluation Metrics
def calculate_metrics(y_true, y_pred):
    print("Calculating metrics")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    r2 = r2_score(y_true, y_pred)
    if len(y_true) > 1:
        actual_directions = np.sign(np.diff(y_true))
        predicted_directions = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(actual_directions == predicted_directions) * 100
    else:
        directional_accuracy = 0
    return rmse, mae, mape, r2, directional_accuracy

# Basic U.S. stock market holiday list for 2025 (expand as needed)
def get_us_holidays():
    holidays = [
        datetime(2025, 1, 1),   # New Year's Day
        datetime(2025, 7, 4),   # Independence Day
        datetime(2025, 11, 27), # Thanksgiving (last Thursday in November)
        datetime(2025, 12, 25), # Christmas
    ]
    return holidays

# Main function
def main(stock='AAPL', term='short'):
    print(f"Starting main for stock={stock}, term={term}")
    try:
        if term == 'short':
            forecast_days = 5
            model_type = 'prophet'
            print(f"Selected short term: {forecast_days} days with Prophet")
        elif term == 'long':
            forecast_days = 20
            model_type = 'gru'
            print(f"Selected long term: {forecast_days} days with GRU")
        else:
            raise ValueError("Invalid term. Use 'short' or 'long'.")
        
        data = fetch_stock_data(stock)
        engineered_data = engineer_features(data)
        
        if model_type == 'prophet':
            prophet_data = prepare_prophet_data(engineered_data, with_features=True)
            test_preds, future_preds, y_test, test_dates, future_dates = train_prophet_model(
                prophet_data, forecast_days=forecast_days, with_features=True
            )
            metrics = calculate_metrics(y_test, test_preds)
            result = {
                'stock': stock,
                'term': term,
                'metrics': [{'Model': 'Prophet with Features', 'RMSE': metrics[0], 'MAE': metrics[1], 'MAPE (%)': metrics[2], 'R²': metrics[3], 'Directional Accuracy (%)': metrics[4]}],
                'future_predictions': future_preds.tolist(),
                'future_dates': [d.strftime('%Y-%m-%d') for d in future_dates]
            }
        elif model_type == 'gru':
            X_train, X_test, y_train, y_test, scaled_data, scaler = prepare_gru_data(
                engineered_data, with_features=True
            )
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            test_preds = predict_gru_test(gru_model, X_test, gru_scaler, with_features=True)
            y_test_actual = inverse_transform_y(y_test, gru_scaler, with_features=True)
            metrics = calculate_metrics(y_test_actual, test_preds)
            last_sequence = scaled_data[-90:]
            future_preds, future_dates = predict_gru_future(gru_model, last_sequence, gru_scaler, engineered_data, forecast_days=forecast_days, with_features=True)
            result = {
                'stock': stock,
                'term': term,
                'metrics': [{'Model': 'GRU with Features', 'RMSE': metrics[0], 'MAE': metrics[1], 'MAPE (%)': metrics[2], 'R²': metrics[3], 'Directional Accuracy (%)': metrics[4]}],
                'future_predictions': future_preds,
                'future_dates': [d.strftime('%Y-%m-%d') for d in future_dates]
            }
        
        print(f"Returning result for {term} term with {len(future_preds)} predictions")
        return result
    except Exception as e:
        print(f"Error in main: {str(e)}")
        print("Traceback:", traceback.format_exc())
        raise

@app.route('/api/forecast')
def forecast():
    stock = request.args.get('stock', 'AAPL')
    term = request.args.get('term', 'short')
    try:
        print(f"Processing request for stock={stock}, term={term}")
        result = main(stock, term)
        print("Successfully generated result:", result)
        return jsonify(result)
    except Exception as e:
        error_msg = f"Error in forecast: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)