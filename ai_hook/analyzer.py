import numpy as np
import pandas as pd
import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

# Technical analysis
import talib
from talib import abstract

# For pattern detection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Utilities
from utils.logger import logger

class MarketAnalyzer:
    """
    Advanced market analyzer that processes real-time market data,
    applies technical indicators, detects patterns, and generates trading signals.
    """
    
    def __init__(self):
        """Initialize the market analyzer with necessary components"""
        # Configure paths
        self.models_dir = "ai_hook/models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Historical data storage
        self.data_store = {}  # Symbol -> DataFrame mapping
        self.predictions_history = {}  # Store past predictions for accuracy tracking
        
        # Performance tracking
        self.accuracy = {}  # Symbol -> accuracy score mapping
        self.signals_count = 0
        self.correct_signals = 0
        
        # Pattern detection confidence thresholds
        self.min_confidence = 0.90  # 90% confidence threshold for sending signals
        self.volatility_threshold = 0.03  # 3% volatility threshold to avoid high volatility periods
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Market Analyzer initialized")
    
    def _initialize_models(self):
        """Initialize or load ML models for pattern detection"""
        self.models = {}
        self.scalers = {}
        
        # Check if pre-trained models exist
        model_path = os.path.join(self.models_dir, "pattern_detector.joblib")
        scaler_path = os.path.join(self.models_dir, "feature_scaler.joblib")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load existing models
            try:
                self.models["pattern_detector"] = joblib.load(model_path)
                self.scalers["pattern_detector"] = joblib.load(scaler_path)
                logger.info("Loaded existing pattern detection model")
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                self._create_new_models()
        else:
            # Create new models
            self._create_new_models()
    
    def _create_new_models(self):
        """Create new models for pattern detection"""
        # Pattern detector model (Random Forest for classification)
        self.models["pattern_detector"] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Feature scaler
        self.scalers["pattern_detector"] = StandardScaler()
        
        logger.info("Created new pattern detection model")
    
    async def process_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process new market data for a symbol and generate analysis
        
        Args:
            symbol: Trading pair symbol
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with analysis results
        """
        # Store data for the symbol
        if symbol in self.data_store:
            # Append new data, avoiding duplicates
            last_timestamp = self.data_store[symbol].index[-1] if not self.data_store[symbol].empty else None
            if last_timestamp and data.index[-1] > last_timestamp:
                # Filter to only new data
                new_data = data[data.index > last_timestamp]
                self.data_store[symbol] = pd.concat([self.data_store[symbol], new_data])
            else:
                self.data_store[symbol] = data
        else:
            self.data_store[symbol] = data
        
        # Ensure we have enough data for analysis
        if len(self.data_store[symbol]) < 50:
            return {"symbol": symbol, "signal": None, "confidence": 0, "message": "Insufficient data for analysis"}
        
        # Apply technical indicators
        df = self._apply_indicators(self.data_store[symbol].copy())
        
        # Check for high volatility periods
        is_high_volatility = self._check_volatility(df)
        if is_high_volatility:
            return {
                "symbol": symbol,
                "signal": None,
                "confidence": 0,
                "message": "High volatility detected, avoiding signal generation"
            }
        
        # Detect patterns
        patterns = self._detect_patterns(df)
        
        # Generate signal if confidence is high enough
        if patterns["confidence"] >= self.min_confidence:
            # Get suggested entry price
            entry_price = df.iloc[-1]["close"]
            
            # Get main reason for signal
            reason = patterns["pattern"] if patterns["pattern"] else "Technical indicator confluence"
            
            signal = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "signal": patterns["direction"],  # "UP" or "DOWN"
                "confidence": patterns["confidence"],
                "entry_price": entry_price,
                "reason": reason,
                "indicators": {
                    "rsi": df.iloc[-1]["rsi"],
                    "macd": df.iloc[-1]["macd"],
                    "ema_crossover": patterns.get("ema_crossover", False),
                    "bb_signal": patterns.get("bb_signal", None)
                }
            }
            
            # Store prediction for later accuracy assessment
            prediction_id = f"{symbol}_{int(time.time())}"
            self.predictions_history[prediction_id] = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "direction": patterns["direction"],
                "entry_price": entry_price,
                "confidence": patterns["confidence"],
                "validated": False,
                "was_correct": None
            }
            
            return signal
        
        return {
            "symbol": symbol,
            "signal": None,
            "confidence": patterns["confidence"],
            "message": "No high-confidence signal detected"
        }
    
    def _apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply technical indicators to the price data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Ensure all required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            return df
        
        # Convert to numpy arrays for talib
        open_data = df["open"].values
        high_data = df["high"].values
        low_data = df["low"].values
        close_data = df["close"].values
        volume_data = df["volume"].values if "volume" in df.columns else None
        
        # RSI (Relative Strength Index)
        df["rsi"] = talib.RSI(close_data, timeperiod=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = talib.MACD(
            close_data, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        
        # Exponential Moving Averages
        df["ema_9"] = talib.EMA(close_data, timeperiod=9)
        df["ema_21"] = talib.EMA(close_data, timeperiod=21)
        df["ema_50"] = talib.EMA(close_data, timeperiod=50)
        df["ema_200"] = talib.EMA(close_data, timeperiod=200)
        
        # Average True Range (ATR) for volatility
        df["atr"] = talib.ATR(high_data, low_data, close_data, timeperiod=14)
        
        # Stochastic Oscillator
        df["slowk"], df["slowd"] = talib.STOCH(
            high_data, low_data, close_data, 
            fastk_period=14, slowk_period=3, slowk_matype=0, 
            slowd_period=3, slowd_matype=0
        )
        
        # Pattern recognition (using talib built-in patterns)
        # Head and Shoulders
        df["cdl_head_shoulders"] = talib.CDLHSANDSHOULDERS(
            open_data, high_data, low_data, close_data
        )
        
        # Double Top/Bottom
        df["cdl_double_top"] = talib.CDLDOUBLECROW(
            open_data, high_data, low_data, close_data
        )
        
        # Engulfing pattern
        df["cdl_engulfing"] = talib.CDLENGULFING(
            open_data, high_data, low_data, close_data
        )
        
        # Add more candlestick patterns as needed
        
        # Drop NaN values that result from calculations
        df = df.dropna()
        
        return df
    
    def _check_volatility(self, df: pd.DataFrame) -> bool:
        """
        Check if the market is experiencing high volatility
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            True if high volatility detected, False otherwise
        """
        if "atr" not in df.columns or df.empty:
            return False
        
        # Calculate ATR as percentage of price
        recent_df = df.tail(10)  # Look at recent price action
        atr_percentage = recent_df["atr"] / recent_df["close"]
        
        # Check if recent ATR is above threshold
        return atr_percentage.mean() > self.volatility_threshold
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect chart patterns and generate trading signals
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with pattern detection results
        """
        result = {
            "direction": None,  # "UP" or "DOWN"
            "confidence": 0.0,
            "pattern": None
        }
        
        if df.empty or len(df) < 30:
            return result
        
        # Focus on recent price action
        recent_df = df.tail(30)
        
        # Check pattern indicators
        patterns_detected = []
        
        # 1. Check for RSI conditions
        rsi = recent_df.iloc[-1]["rsi"]
        if rsi < 30:
            patterns_detected.append(("RSI oversold", 0.6, "UP"))
        elif rsi > 70:
            patterns_detected.append(("RSI overbought", 0.6, "DOWN"))
        
        # 2. Check for MACD crossover
        if recent_df.iloc[-2]["macd"] < recent_df.iloc[-2]["macd_signal"] and \
           recent_df.iloc[-1]["macd"] > recent_df.iloc[-1]["macd_signal"]:
            patterns_detected.append(("MACD bullish crossover", 0.65, "UP"))
        elif recent_df.iloc[-2]["macd"] > recent_df.iloc[-2]["macd_signal"] and \
             recent_df.iloc[-1]["macd"] < recent_df.iloc[-1]["macd_signal"]:
            patterns_detected.append(("MACD bearish crossover", 0.65, "DOWN"))
        
        # 3. Check for Bollinger Bands squeeze and breakout
        bb_width = (recent_df["bb_upper"] - recent_df["bb_lower"]) / recent_df["bb_middle"]
        bb_squeeze = bb_width.rolling(window=10).mean().iloc[-1] < bb_width.rolling(window=20).mean().iloc[-1]
        
        if bb_squeeze and recent_df.iloc[-1]["close"] > recent_df.iloc[-1]["bb_upper"]:
            patterns_detected.append(("Bollinger Band breakout (up)", 0.7, "UP"))
        elif bb_squeeze and recent_df.iloc[-1]["close"] < recent_df.iloc[-1]["bb_lower"]:
            patterns_detected.append(("Bollinger Band breakout (down)", 0.7, "DOWN"))
        
        # 4. Check for EMA crossovers
        if recent_df.iloc[-2]["ema_9"] < recent_df.iloc[-2]["ema_21"] and \
           recent_df.iloc[-1]["ema_9"] > recent_df.iloc[-1]["ema_21"]:
            patterns_detected.append(("EMA 9/21 bullish crossover", 0.6, "UP"))
            result["ema_crossover"] = True
        elif recent_df.iloc[-2]["ema_9"] > recent_df.iloc[-2]["ema_21"] and \
             recent_df.iloc[-1]["ema_9"] < recent_df.iloc[-1]["ema_21"]:
            patterns_detected.append(("EMA 9/21 bearish crossover", 0.6, "DOWN"))
            result["ema_crossover"] = True
        
        # 5. Check for candlestick patterns
        if recent_df.iloc[-1]["cdl_engulfing"] > 0:
            patterns_detected.append(("Bullish engulfing pattern", 0.75, "UP"))
        elif recent_df.iloc[-1]["cdl_engulfing"] < 0:
            patterns_detected.append(("Bearish engulfing pattern", 0.75, "DOWN"))
        
        # 6. Check for Head and Shoulders pattern
        if recent_df.iloc[-1]["cdl_head_shoulders"] != 0:
            direction = "UP" if recent_df.iloc[-1]["cdl_head_shoulders"] > 0 else "DOWN"
            patterns_detected.append((f"Head and shoulders pattern ({direction.lower()})", 0.8, direction))
        
        # If no patterns detected, return default result
        if not patterns_detected:
            return result
        
        # Calculate overall direction and confidence based on detected patterns
        up_confidence = sum([conf for _, conf, dir in patterns_detected if dir == "UP"])
        down_confidence = sum([conf for _, conf, dir in patterns_detected if dir == "DOWN"])
        
        # Apply machine learning model for additional confidence if available
        ml_confidence, ml_direction = self._apply_ml_prediction(df)
        
        if ml_confidence > 0:
            if ml_direction == "UP":
                up_confidence += ml_confidence
            else:
                down_confidence += ml_confidence
        
        # Determine overall direction and confidence
        if up_confidence > down_confidence:
            patterns = [pattern for pattern, _, dir in patterns_detected if dir == "UP"]
            result["direction"] = "UP"
            result["confidence"] = min(0.99, up_confidence / (up_confidence + down_confidence))
            result["pattern"] = ", ".join(patterns) if patterns else None
        elif down_confidence > up_confidence:
            patterns = [pattern for pattern, _, dir in patterns_detected if dir == "DOWN"]
            result["direction"] = "DOWN"
            result["confidence"] = min(0.99, down_confidence / (up_confidence + down_confidence))
            result["pattern"] = ", ".join(patterns) if patterns else None
        
        # Extra confidence boost if multiple patterns agree
        if len(patterns_detected) >= 3 and (all(dir == "UP" for _, _, dir in patterns_detected) or 
                                           all(dir == "DOWN" for _, _, dir in patterns_detected)):
            result["confidence"] = min(0.99, result["confidence"] + 0.15)
        
        return result
    
    def _apply_ml_prediction(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Apply machine learning model to predict price movement
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Tuple of (confidence, direction)
        """
        if "pattern_detector" not in self.models:
            return 0.0, None
        
        # If model hasn't been trained with data yet, return no confidence
        model = self.models["pattern_detector"]
        if not hasattr(model, 'classes_'):
            return 0.0, None
        
        try:
            # Extract features for prediction
            features = self._extract_features(df)
            
            # Scale features
            scaler = self.scalers["pattern_detector"]
            if not hasattr(scaler, 'n_features_in_'):
                # Scaler not fitted yet
                return 0.0, None
            
            scaled_features = scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction_proba = model.predict_proba(scaled_features)[0]
            prediction_idx = prediction_proba.argmax()
            confidence = prediction_proba[prediction_idx]
            direction = "UP" if model.classes_[prediction_idx] == 1 else "DOWN"
            
            return confidence, direction
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {str(e)}")
            return 0.0, None
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for machine learning prediction
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            NumPy array of features
        """
        # Use recent data
        recent_df = df.tail(20)
        
        # Extract relevant features
        features = []
        
        # RSI
        features.append(recent_df.iloc[-1]["rsi"])
        
        # MACD
        features.append(recent_df.iloc[-1]["macd"])
        features.append(recent_df.iloc[-1]["macd_signal"])
        features.append(recent_df.iloc[-1]["macd_hist"])
        
        # Bollinger Bands
        close = recent_df.iloc[-1]["close"]
        bb_upper = recent_df.iloc[-1]["bb_upper"]
        bb_lower = recent_df.iloc[-1]["bb_lower"]
        bb_middle = recent_df.iloc[-1]["bb_middle"]
        
        features.append((close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5)
        features.append((close - bb_middle) / bb_middle if bb_middle != 0 else 0)
        
        # EMA positions
        features.append((close - recent_df.iloc[-1]["ema_9"]) / close if close != 0 else 0)
        features.append((close - recent_df.iloc[-1]["ema_21"]) / close if close != 0 else 0)
        features.append((close - recent_df.iloc[-1]["ema_50"]) / close if close != 0 else 0)
        
        # Price momentum
        price_changes = recent_df["close"].pct_change().dropna().values
        features.append(price_changes.mean())
        features.append(price_changes.std())
        
        # Volume changes
        if "volume" in recent_df.columns:
            volume_changes = recent_df["volume"].pct_change().dropna().values
            features.append(volume_changes.mean())
        else:
            features.append(0)
        
        return np.array(features)
    
    async def update_prediction_result(self, prediction_id: str, price: float) -> bool:
        """
        Update a prediction with actual result to track accuracy
        
        Args:
            prediction_id: ID of the prediction to update
            price: Current price to evaluate the prediction
            
        Returns:
            True if prediction was successful, False otherwise
        """
        if prediction_id not in self.predictions_history:
            return False
        
        prediction = self.predictions_history[prediction_id]
        if prediction["validated"]:
            return prediction["was_correct"]
        
        # Get the entry price
        entry_price = prediction["entry_price"]
        
        # Compare current price with entry price
        if prediction["direction"] == "UP":
            was_correct = price > entry_price
        else:  # direction is "DOWN"
            was_correct = price < entry_price
        
        # Update prediction history
        prediction["validated"] = True
        prediction["was_correct"] = was_correct
        prediction["validation_time"] = datetime.now()
        prediction["validation_price"] = price
        
        # Update accuracy stats
        symbol = prediction["symbol"]
        if symbol not in self.accuracy:
            self.accuracy[symbol] = {"correct": 0, "total": 0}
        
        self.accuracy[symbol]["total"] += 1
        if was_correct:
            self.accuracy[symbol]["correct"] += 1
            self.correct_signals += 1
        
        self.signals_count += 1
        
        # Update model if necessary
        await self._update_model(prediction)
        
        return was_correct
    
    async def _update_model(self, prediction: Dict[str, Any]) -> None:
        """
        Update the ML model based on prediction results
        
        Args:
            prediction: Dictionary with prediction details
        """
        # Implementation for updating the model would go here
        # This would typically involve collecting enough samples and retraining periodically
        pass
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """
        Get accuracy statistics for predictions
        
        Returns:
            Dictionary with accuracy statistics
        """
        stats = {
            "overall": 0.0,
            "by_symbol": {},
            "total_signals": self.signals_count,
            "correct_signals": self.correct_signals
        }
        
        # Calculate overall accuracy
        if self.signals_count > 0:
            stats["overall"] = self.correct_signals / self.signals_count
        
        # Calculate accuracy by symbol
        for symbol, acc in self.accuracy.items():
            if acc["total"] > 0:
                stats["by_symbol"][symbol] = acc["correct"] / acc["total"]
            else:
                stats["by_symbol"][symbol] = 0.0
        
        return stats
    
    def save_models(self) -> None:
        """Save trained models to disk"""
        if "pattern_detector" in self.models and hasattr(self.models["pattern_detector"], 'classes_'):
            joblib.dump(self.models["pattern_detector"], 
                       os.path.join(self.models_dir, "pattern_detector.joblib"))
            
            joblib.dump(self.scalers["pattern_detector"], 
                      os.path.join(self.models_dir, "feature_scaler.joblib"))
            
            logger.info("Saved pattern detection models")
    
    def train_with_historical_data(self, symbol: str, data: pd.DataFrame, labels: pd.Series) -> None:
        """
        Train the ML model with historical data and known outcomes
        
        Args:
            symbol: Trading pair symbol
            data: DataFrame with OHLCV data
            labels: Series with known outcomes (1 for UP, 0 for DOWN)
        """
        if data.empty or len(labels) == 0:
            logger.warning("Cannot train with empty data")
            return
        
        try:
            # Apply indicators
            df = self._apply_indicators(data.copy())
            
            # Extract features for each data point
            features = []
            valid_indices = []
            
            for i in range(len(df) - 20):  # Need at least 20 data points for feature extraction
                try:
                    feature_vector = self._extract_features(df.iloc[i:i+20])
                    features.append(feature_vector)
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(f"Error extracting features for index {i}: {str(e)}")
            
            if not features:
                logger.warning("No valid features extracted for training")
                return
            
            features = np.array(features)
            
            # Get corresponding labels
            valid_labels = labels.iloc[valid_indices].values
            
            # Fit the scaler
            self.scalers["pattern_detector"].fit(features)
            
            # Scale features
            scaled_features = self.scalers["pattern_detector"].transform(features)
            
            # Train the model
            self.models["pattern_detector"].fit(scaled_features, valid_labels)
            
            logger.info(f"Trained model for {symbol} with {len(valid_labels)} samples")
            
            # Save the trained model
            self.save_models()
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}") 