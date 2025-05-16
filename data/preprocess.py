import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from utils.config import ModelConfig
from utils.logger import logger

class DataPreprocessor:
    """Handles data preprocessing for time series forecasting"""
    
    def __init__(self):
        self.window_size = ModelConfig.WINDOW_SIZE
        self.scalers = {}
    
    def add_features(self, df):
        """Add technical indicators to the dataframe"""
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Target: price direction (1 if price goes up, 0 if it goes down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop rows with NaN values
        return df.dropna()
    
    def normalize_data(self, df, symbol, is_training=True):
        """Normalize the data using Min-Max scaling"""
        features = [col for col in df.columns if col != 'target']
        
        if is_training:
            scaler = MinMaxScaler()
            df[features] = scaler.fit_transform(df[features])
            self.scalers[symbol] = scaler
        else:
            if symbol in self.scalers:
                df[features] = self.scalers[symbol].transform(df[features])
            else:
                logger.warning(f"No scaler found for {symbol}. Using raw data.")
        
        return df
    
    def create_windows(self, data):
        """Create sliding windows for time series data"""
        X, y = [], []
        
        for i in range(len(data) - self.window_size):
            # Get window of features (all columns except target)
            features = data.iloc[i:(i + self.window_size), :-1].values
            X.append(features)
            
            # Get the target value
            target = data.iloc[i + self.window_size, -1]
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, symbol, is_training=True):
        """Prepare data for model training or prediction"""
        try:
            # Add features and clean data
            df_processed = self.add_features(df)
            
            if len(df_processed) <= self.window_size:
                logger.warning(f"Not enough data for {symbol}. Need more than {self.window_size} points.")
                return np.array([]), np.array([])
            
            # Normalize data
            df_normalized = self.normalize_data(df_processed, symbol, is_training)
            
            # Create windows
            X, y = self.create_windows(df_normalized)
            
            logger.info(f"Prepared {len(X)} samples for {symbol}")
            return X, y
        
        except Exception as e:
            logger.error(f"Error preprocessing data for {symbol}: {str(e)}")
            return np.array([]), np.array([])
    
    def prepare_latest_data(self, df, symbol):
        """Prepare the latest data for prediction"""
        try:
            # Add features and clean data
            df_processed = self.add_features(df)
            
            if len(df_processed) < self.window_size:
                logger.warning(f"Not enough latest data for {symbol}")
                return None
            
            # Normalize using existing scaler
            df_normalized = self.normalize_data(df_processed, symbol, False)
            
            # Get the last window
            latest_window = df_normalized.iloc[-self.window_size:, :-1].values
            
            # Reshape for model input (batch_size, time_steps, features)
            return np.array([latest_window])
        
        except Exception as e:
            logger.error(f"Error preparing latest data for {symbol}: {str(e)}")
            return None
    
    def save_scalers(self, directory="saved_models"):
        """Save scalers to disk"""
        os.makedirs(directory, exist_ok=True)
        
        for symbol, scaler in self.scalers.items():
            filename = os.path.join(directory, f"scaler_{symbol.replace('/', '_')}.pkl")
            joblib.dump(scaler, filename)
    
    def load_scalers(self, directory="saved_models"):
        """Load scalers from disk"""
        if not os.path.exists(directory):
            return
        
        for file in os.listdir(directory):
            if file.startswith("scaler_") and file.endswith(".pkl"):
                symbol = file[7:-4].replace('_', '/')
                self.scalers[symbol] = joblib.load(os.path.join(directory, file))

