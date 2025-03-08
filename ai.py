import pandas as pd
import numpy as np
import time
import os
import ccxt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import psutil
import argparse
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.panel import Panel
import requests
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from tensorflow.keras.layers import GRU, Bidirectional
from sklearn.metrics import precision_score, recall_score, f1_score
import subprocess
import sys

# Initialize rich console
console = Console()

def install_requirements():
    """Auto install missing packages"""
    required_packages = [
        'pandas',
        'numpy',
        'ccxt',
        'scikit-learn',
        'tensorflow',
        'psutil',
        'rich',
        'requests',
        'ta'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_environment():
    """Setup required directories and files"""
    # Create directories
    for dir_name in ['data', 'models', 'logs']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
    # Create empty data files if needed
    data_files = {
        'data/historical_data.csv': 'Timestamp,Open,High,Low,Close,Volume\n',
        'models/model_metadata.json': '{"last_training": null, "best_accuracy": 0}\n'
    }
    
    for file_path, header in data_files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(header)

# Tính RSI
def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

# Tải dữ liệu từ Binance
def fetch_historical_data(timeframe='5m', limit=10000, symbol='BTC/USDT'):
    """Fetch extended historical data with proper ordering"""
    try:
        exchange = ccxt.binance()
        all_data = []
        
        # Calculate how many requests we need
        max_limit = 1000  # Binance maximum limit per request
        num_requests = (limit + max_limit - 1) // max_limit
        
        # Get initial timestamp for the oldest data
        now = exchange.milliseconds()
        since = now - (limit * 5 * 60 * 1000)  # Convert limit to milliseconds (5min timeframe)
        
        print(f"🔄 Đang tải {limit} mẫu dữ liệu...")
        
        for i in range(num_requests):
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since + (i * max_limit * 5 * 60 * 1000),
                limit=min(max_limit, limit - i * max_limit)
            )
            all_data.extend(ohlcv)
            print(f"✓ Đã tải {len(all_data)}/{limit} mẫu")
            time.sleep(1)  # Avoid rate limits
        
        # Convert to DataFrame and process
        data = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
        data.set_index('Timestamp', inplace=True)
        data = data.sort_index()  # Ensure chronological order
        
        # Add technical indicators
        data['RSI'] = calculate_rsi(data)
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['VOL_MA20'] = data['Volume'].rolling(window=20).mean()
        
        # Clean data
        data = data.dropna()  # Remove missing values
        data = data[data['Volume'] > 0]  # Remove zero volume periods
        
        print(f"✅ Đã tải xong {len(data)} mẫu dữ liệu từ {data.index[0]} đến {data.index[-1]}")
        return data

    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None

def load_or_fetch_historical_data(timeframe='5m', total_samples=100000, symbol='DOGE/USDT', data_file='doge_historical.csv'):
    """Load data from file or fetch if not available"""
    try:
        # Check if data file exists and is recent
        if os.path.exists(data_file):
            data = pd.read_csv(data_file)
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            data.set_index('Timestamp', inplace=True)
            
            # Check if we have enough recent data
            if len(data) >= total_samples:
                latest_time = data.index.max()
                current_time = pd.Timestamp.now()
                if (current_time - latest_time).total_seconds() < 3600:  # Data is less than 1 hour old
                    print(f"✅ Loaded {len(data)} samples from existing file")
                    return data
        
        print("🔄 Fetching new historical data...")
        exchange = ccxt.binance()
        all_data = []
        
        # Calculate how many requests we need (1000 samples per request)
        max_limit = 1000
        num_requests = (total_samples + max_limit - 1) // max_limit
        
        # Get initial timestamp for the oldest data
        now = exchange.milliseconds()
        since = now - (total_samples * 5 * 60 * 1000)  # Convert to milliseconds
        
        for i in range(num_requests):
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since + (i * max_limit * 5 * 60 * 1000),
                    limit=min(max_limit, total_samples - i * max_limit)
                )
                all_data.extend(ohlcv)
                print(f"✓ Fetched {len(all_data)}/{total_samples} samples")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"⚠️ Error in batch {i+1}: {str(e)}")
                time.sleep(5)
                continue
        
        # Convert to DataFrame
        data = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms')
        data.set_index('Timestamp', inplace=True)
        data = data.sort_index()
        
        # Save to file
        data.to_csv(data_file)
        print(f"✅ Saved {len(data)} samples to {data_file}")
        
        return data

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

# Chuẩn bị dữ liệu
def preprocess_data(data):
    """Enhanced preprocessing for DOGE meme coin"""
    # Add standard technical indicators
    bb = BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_lower'] = bb.bollinger_lband()
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['Close']
    
    # Add momentum indicators
    data['RSI'] = calculate_rsi(data)
    data['ROC'] = data['Close'].pct_change(10)  # Rate of change
    
    # Add volatility indicators
    data['ATR'] = data['High'] - data['Low']
    data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
    
    # Volume analysis
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_STD'] = data['Volume'].rolling(window=20).std()
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Price changes at different timeframes
    for period in [5, 15, 30, 60]:
        data[f'Price_Change_{period}'] = data['Close'].pct_change(period)
    
    # Create features for model
    features = [
        'Close', 'Volume', 'RSI', 'ROC',
        'BB_upper', 'BB_lower', 'BB_width',
        'ATR', 'Volatility',
        'Volume_MA', 'Volume_STD', 'Volume_Change'
    ] + [f'Price_Change_{p}' for p in [5, 15, 30, 60]]
    
    # Use RobustScaler for better handling of DOGE's high volatility
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data[features].fillna(0).values)
    
    return scaled_data, scaler, features

# Tạo dataset cho LSTM
def create_trend_dataset(data, time_step=100, price_threshold=0.005):
    """Enhanced dataset creation with price threshold"""
    X, Y = [], []
    
    for i in range(len(data) - time_step - 1):
        # Input sequence
        x_sequence = data[i:(i + time_step), :]
        
        # Calculate return
        future_return = (data[i + time_step, 0] - data[i + time_step - 1, 0]) / data[i + time_step - 1, 0]
        
        # Label with threshold
        trend = 1 if future_return > price_threshold else (0 if future_return < -price_threshold else -1)
        
        if trend != -1:  # Only add clear trend signals
            X.append(x_sequence)
            Y.append(trend)
    
    return np.array(X), np.array(Y)

def create_model(input_shape, learning_rate=0.001):
    """Enhanced model for DOGE's high volatility"""
    model = Sequential([
        # Deeper network for capturing complex patterns
        Bidirectional(LSTM(512, return_sequences=True, input_shape=input_shape)),
        BatchNormalization(),
        Dropout(0.4),
        
        Bidirectional(LSTM(256, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.4),
        
        Bidirectional(LSTM(128)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # More aggressive optimizer for DOGE's volatility
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Enhanced model evaluation with multiple metrics"""
    predictions_prob = model.predict(X_test)
    predictions = (predictions_prob > threshold).astype(int)
    
    metrics = {
        'accuracy': np.mean(predictions.flatten() == y_test),
        'precision': precision_score(y_test, predictions.flatten()),
        'recall': recall_score(y_test, predictions.flatten()),
        'f1': f1_score(y_test, predictions.flatten())
    }
    
    return metrics, predictions_prob

# Huấn luyện mô hình
def train_model(X_train, y_train, model=None):
    if model is None:
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))  # Tăng neuron lên 256
        model.add(Dropout(0.3))
        model.add(LSTM(128))  # Tăng neuron lên 128
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Tăng patience lên 15
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)  # Tăng patience lên 10
    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stop, reduce_lr])
    return model

# Đánh giá và lưu mô hình
def evaluate_and_save_model(model, X_test, y_test, current_best_accuracy):
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions.flatten() == y_test)
    if accuracy > current_best_accuracy:
        model.save('btc_trend_model.keras')  # Dùng định dạng .keras
        print(f"Đã lưu mô hình mới với độ chính xác: {accuracy:.2%}")
        return accuracy
    else:
        print(f"Mô hình mới có độ chính xác {accuracy:.2%}, không tốt hơn mô hình hiện tại ({current_best_accuracy:.2%})")
        return current_best_accuracy

def get_combined_data(historical_data=None, limit=10000):
    """Get and combine historical data with proper ordering"""
    try:
        # Get new data
        new_data = fetch_historical_data(limit=limit)
        if (new_data is None) or new_data.empty:
            return historical_data
            
        # If no historical data exists, return new data
        if historical_data is None:
            return new_data
            
        # Combine and remove duplicates, keeping proper order
        combined_data = pd.concat([historical_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        # Keep most recent samples up to limit
        return combined_data.tail(limit)
        
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật dữ liệu: {str(e)}")
        return historical_data

def limit_cpu_usage(cpu_limit=80):
    """Hạn chế sử dụng CPU"""
    process = psutil.Process()
    process.cpu_percent(interval=1)  # Reset CPU usage counter
    if process.cpu_percent() > cpu_limit:  # Giới hạn CPU
        time.sleep(0.1)  # Delay để giảm tải CPU

def update_training_metrics(metrics):
    try:
        requests.post("http://localhost:8000/update_metrics", json=metrics)
    except:
        pass  # Silently fail if web server is not running

def train_with_time_limit(hours, cpu_limit=95, initial_accuracy=0):
    """Modified training for DOGE"""
    print(f"🕒 Starting DOGE training for {hours} hours...")
    end_time = datetime.now() + timedelta(hours=hours)
    best_accuracy = initial_accuracy
    consecutive_fails = 0
    max_fails = 5
    model_path = 'doge_trend_model.keras'
    best_model_path = 'doge_best_model.keras'
    model = None  # Initialize model variable
    start_time = datetime.now()

    # Load or fetch historical data
    historical_data = load_or_fetch_historical_data(
        timeframe='5m',
        total_samples=100000,
        symbol='DOGE/USDT',
        data_file='doge_historical.csv'
    )
    
    if historical_data is None:
        print("❌ Could not load or fetch DOGE data")
        return 0

    # Process initial data to get input shape
    scaled_data, scaler, _ = preprocess_data(historical_data)
    X, y = create_trend_dataset(scaled_data)
    input_shape = (X.shape[1], X.shape[2])

    # Load existing model if available
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print("✅ Đã tải model cũ thành công")
            # Evaluate existing model
            train_size = int(len(X) * 0.8)
            _, X_test = X[:train_size], X[train_size:]
            _, y_test = y[:train_size], y[train_size:]
            predictions = model.predict(X_test, verbose=0)
            best_accuracy = np.mean((predictions > 0.5).astype(int).flatten() == y_test)
            print(f"📊 Độ chính xác model cũ: {best_accuracy:.2%}")
            # Save a backup of the old model
            model.save(f'doge_backup_{datetime.now().strftime("%Y%m%d_%H%M")}_model.keras')
        except Exception as e:
            print(f"⚠️ Không thể tải model cũ: {str(e)}")
            model = None
            best_accuracy = 0
    
    # Create new model if none exists
    if model is None:
        print("🔄 Tạo model mới...")
        model = create_model(input_shape)

    while datetime.now() < end_time:
        try:
            # Get fresh data
            historical_data = get_combined_data(historical_data)
            if historical_data is None or historical_data.empty:
                print("❌ Không thể lấy dữ liệu mới")
                time.sleep(60)
                continue

            # Preprocess data
            scaled_data, scaler, _ = preprocess_data(historical_data)
            X, y = create_trend_dataset(scaled_data)
            
            if len(X) < 100:
                print("⚠️ Chưa đủ dữ liệu để huấn luyện")
                time.sleep(60)
                continue

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=64,
                epochs=50,
                validation_split=0.2,
                verbose=1
            )

            # Evaluate model
            predictions = model.predict(X_test, verbose=0)
            current_accuracy = np.mean((predictions > 0.5).astype(int).flatten() == y_test)

            # Update metrics
            metrics = {
                "accuracy": current_accuracy,
                "best_accuracy": best_accuracy,
                "epoch": 50,
                "loss": history.history['loss'][-1],
                "training_time": str(datetime.now() - start_time).split('.')[0],
                "epochs_history": list(range(1, 51)),
                "accuracy_history": [float(acc) for acc in history.history['accuracy']],
                "validation_accuracy_history": [float(acc) for acc in history.history['val_accuracy']]
            }
            update_training_metrics(metrics)

            # Save if better
            if current_accuracy > best_accuracy:
                print(f"✨ New best accuracy: {current_accuracy:.2%}")
                model.save(best_model_path)
                if os.path.exists(best_model_path):
                    os.replace(best_model_path, model_path)
                best_accuracy = current_accuracy
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                if consecutive_fails >= max_fails:
                    print("⚠️ Model not improving, resetting...")
                    model = create_model(input_shape)
                    consecutive_fails = 0

            # CPU control and rest
            limit_cpu_usage(cpu_limit)
            time.sleep(2)

        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            time.sleep(60)

    return best_accuracy

def train_with_attempts(max_attempts, target_accuracy, cpu_limit=80):
    print(f"🕒 Bắt đầu huấn luyện với tối đa {max_attempts} lần thử...")
    model_path = 'btc_trend_model.keras'
    
    # Load existing model if available
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print("✅ Đã tải model cũ thành công")
            # Evaluate existing model
            historical_data = fetch_historical_data()
            scaled_data, scaler, _ = preprocess_data(historical_data)  # Add _ to catch the third return value
            X, y = create_trend_dataset(scaled_data)
            train_size = int(len(X) * 0.8)
            _, X_test = X[:train_size], X[train_size:]
            _, y_test = y[:train_size], y[train_size:]
            predictions = model.predict(X_test, verbose=0)
            best_accuracy = np.mean((predictions > 0.5).astype(int).flatten() == y_test)
            print(f"📊 Độ chính xác model cũ: {best_accuracy:.2%}")
        except Exception as e:
            print(f"⚠️ Không thể tải model cũ: {str(e)}")
            model = None
            best_accuracy = 0
    else:
        print("ℹ️ Không tìm thấy model cũ, sẽ tạo model mới")
        model = None
        best_accuracy = 0

    consecutive_fails = 0
    max_fails = 5  # Số lần thất bại liên tiếp tối đa

    historical_data = None
    attempt = 0
    start_time = datetime.now()
    while best_accuracy < target_accuracy and attempt < max_attempts:
        try:
            # Kiểm soát tài nguyên
            limit_cpu_usage(cpu_limit)
            
            # Cập nhật dữ liệu
            historical_data = get_combined_data(historical_data)
            if historical_data is None or historical_data.empty:
                print("❌ Không thể lấy dữ liệu")
                time.sleep(60)
                continue

            # Xử lý dữ liệu
            scaled_data, scaler, features = preprocess_data(historical_data)  # Properly unpack all three values
            X, y = create_trend_dataset(scaled_data)
            
            if len(X) < 100:
                print("⚠️ Chưa đủ dữ liệu để huấn luyện")
                time.sleep(60)
                continue

            # Chia tập train/test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Huấn luyện với nhiều epochs và kiểm tra sau mỗi epoch
            model = Sequential([
                LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.3),
                LSTM(128),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            for epoch in range(50):  # 50 epochs mỗi lần train
                # Train một epoch
                history = model.fit(X_train, y_train, 
                                  batch_size=32,
                                  epochs=1, 
                                  validation_split=0.2,
                                  verbose=1)
                
                # Đánh giá sau mỗi epoch
                predictions = model.predict(X_test, verbose=0)
                current_accuracy = np.mean((predictions > 0.5).astype(int).flatten() == y_test)
                
                print(f"Epoch {epoch+1}/50 - Accuracy: {current_accuracy:.2%}")
                
                # Kiểm tra xu hướng tăng
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    consecutive_fails = 0
                    print(f"✨ Đã tìm thấy mô hình tốt hơn! Độ chính xác: {best_accuracy:.2%}")
                    model.save('btc_trend_model.keras')
                else:
                    consecutive_fails += 1
                
                # Reset nếu không cải thiện sau nhiều lần
                if consecutive_fails >= max_fails:
                    print("⚠️ Không có cải thiện sau nhiều lần, thử lại với cấu trúc mới...")
                    break  # Thoát để bắt đầu vòng train mới
                
                # Kiểm soát tài nguyên
                limit_cpu_usage(cpu_limit)

                metrics = {
                    "accuracy": current_accuracy,
                    "loss": history.history['loss'][0],
                    "epoch": epoch + 1,
                    "best_accuracy": best_accuracy,
                    "training_time": str(datetime.now() - start_time).split('.')[0],
                    "epochs_history": list(range(epoch + 1)),
                    "accuracy_history": [float(acc) for acc in history.history['accuracy']]
                }
                update_training_metrics(metrics)
                
            print(f"📈 Độ chính xác tốt nhất hiện tại: {best_accuracy:.2%}")
            
            # Nghỉ ngắn trước khi tiếp tục
            time.sleep(2)
            
            attempt += 1
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình train: {str(e)}")
            time.sleep(60)  # Đợi 1 phút trước khi thử lại
            
    return best_accuracy

def cleanup_old_backups(keep_last_n=5):
    """Clean up old model backups, keeping only the N most recent"""
    backups = sorted([f for f in os.listdir('.') if f.startswith('backup_') and f.endswith('_model.keras')])
    if len(backups) > keep_last_n:
        for old_backup in backups[:-keep_last_n]:
            try:
                os.remove(old_backup)
                print(f"Removed old backup: {old_backup}")
            except Exception as e:
                print(f"Could not remove {old_backup}: {e}")

def display_menu():
    """Display the training mode selection menu"""
    console.print(Panel.fit(
        "[bold cyan]BTC Trend Prediction - Training Mode[/bold cyan]\n\n"
        "1. Huấn luyện theo số lần thử\n"
        "2. Huấn luyện theo thời gian\n"
        "3. Thoát",
        title="Menu"
    ))

def get_training_params():
    """Get training parameters from user input"""
    try:
        mode = Prompt.ask("Chọn chế độ", choices=["1", "2", "3"])
        if mode == "3":
            console.print("[yellow]Thoát chương trình...[/yellow]")
            sys.exit(0)

        if mode == "1":
            attempts = IntPrompt.ask("Số lần thử tối đa", default=100)
            target_accuracy = FloatPrompt.ask("Độ chính xác mục tiêu (%)", default=60) / 100
            cpu_limit = IntPrompt.ask("Giới hạn CPU (%)", default=80)
            return {
                'mode': 'attempts',
                'value': attempts,
                'target_accuracy': target_accuracy,
                'cpu_limit': cpu_limit
            }
        else:
            hours = FloatPrompt.ask("Số giờ huấn luyện", default=1)
            cpu_limit = IntPrompt.ask("Giới hạn CPU (%)", default=80)
            return {
                'mode': 'time',
                'value': hours,
                'cpu_limit': cpu_limit
            }
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã hủy bởi người dùng[/yellow]")
        sys.exit(0)

def main():
    try:
        console.print("[bold green]🚀 Khởi động chương trình huấn luyện...[/bold green]")
        
        install_requirements()
        setup_environment()
        
        while True:
            display_menu()
            params = get_training_params()
            
            console.print("\n[bold cyan]Bắt đầu huấn luyện với cấu hình:[/bold cyan]")
            for key, value in params.items():
                if key == 'target_accuracy' and 'target_accuracy' in params:
                    console.print(f"- {key}: {value:.2%}")
                else:
                    console.print(f"- {key}: {value}")
            
            try:
                if params['mode'] == 'time':
                    final_accuracy = train_with_time_limit(
                        params['value'],
                        cpu_limit=params['cpu_limit']
                    )
                else:
                    final_accuracy = train_with_attempts(
                        int(params['value']),
                        params['target_accuracy'],
                        cpu_limit=params['cpu_limit']
                    )
                
                console.print(f"\n[bold green]✅ Huấn luyện hoàn tất![/bold green]")
                console.print(f"[green]Độ chính xác cuối cùng: {final_accuracy:.2%}[/green]")
                
            except Exception as e:
                console.print(f"[bold red]❌ Lỗi trong quá trình huấn luyện: {str(e)}[/bold red]")
            
            if not Prompt.ask("\nTiếp tục huấn luyện?", choices=["y", "n"]) == "y":
                break
                
        console.print("[yellow]👋 Kết thúc chương trình[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã hủy bởi người dùng[/yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Lỗi không mong muốn: {str(e)}[/bold red]")

if __name__ == "__main__":
    main()