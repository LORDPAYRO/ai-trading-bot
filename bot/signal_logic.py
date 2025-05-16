import datetime
import numpy as np
import pandas as pd
import os
import asyncio
from typing import Dict, List, Optional, Union

from data.fetch_data import DataManager
from data.preprocess import DataPreprocessor
from model.lstm_model import LSTMModel
from utils.config import DataConfig, ModelConfig, SignalConfig
from utils.logger import logger

class SignalGenerator:
    """Generates trading signals based on model predictions"""
    
    def __init__(self):
        """Initialize the signal generator"""
        self.data_manager = DataManager()
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.start_time = datetime.datetime.now()
        self.signals_history = []
        self.last_update = None
        
        # Load trained models
        self._load_models()
        logger.info("Signal generator initialized")
    
    def _load_models(self):
        """Load trained models from disk"""
        if not os.path.exists(ModelConfig.MODEL_SAVE_PATH):
            logger.warning(f"Model directory {ModelConfig.MODEL_SAVE_PATH} does not exist")
            return
        
        # Load preprocessor scalers
        self.preprocessor.load_scalers()
        
        # Load models for each symbol
        for filename in os.listdir(ModelConfig.MODEL_SAVE_PATH):
            if filename.startswith("model_") and filename.endswith(".h5"):
                # Extract symbol from filename
                symbol_part = filename[6:-3]  # Remove "model_" prefix and ".h5" suffix
                symbol = symbol_part.replace('_', '/')
                
                try:
                    # Load the model
                    model_path = os.path.join(ModelConfig.MODEL_SAVE_PATH, filename)
                    model = LSTMModel()
                    model.load(model_path)
                    
                    # Store the model
                    self.models[symbol] = model
                    logger.info(f"Loaded model for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error loading model for {symbol}: {str(e)}")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    async def analyze_market(self, symbol: str, timeframe: str = None) -> Optional[Dict]:
        """
        Analyze a specific market and generate a prediction
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval
            
        Returns:
            Dictionary with prediction details or None if prediction fails
        """
        timeframe = timeframe or DataConfig.DEFAULT_TIMEFRAME
        
        try:
            # Check if model is available for this symbol
            if symbol not in self.models:
                logger.warning(f"No model available for {symbol}")
                return None
            
            # Fetch latest data
            df = self.data_manager.fetch_latest_data(symbol, timeframe)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Prepare data for prediction
            X = self.preprocessor.prepare_latest_data(df, symbol)
            
            if X is None or len(X) == 0:
                logger.warning(f"Failed to prepare data for {symbol}")
                return None
            
            # Make prediction with confidence
            model = self.models[symbol]
            prediction, confidence = model.predict_with_confidence(X)
            
            # Extract latest price
            latest_price = df.iloc[-1]['close']
            
            # Get the result
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'prediction': prediction[0][0],  # 1 for UP, 0 for DOWN
                'confidence': confidence[0][0],  # Confidence score
                'timestamp': datetime.datetime.now(),
                'price': latest_price
            }
            
            # Calculate stop loss and take profit
            if SignalConfig.USE_ATR_MULTIPLIER:
                # Calculate ATR (Average True Range)
                high = df['high'].values
                low = df['low'].values
                close = df['close'].values
                
                tr1 = np.abs(high - low)
                tr2 = np.abs(high - np.roll(close, 1))
                tr3 = np.abs(low - np.roll(close, 1))
                
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = np.mean(tr[-SignalConfig.ATR_PERIOD:])
                
                # Set stop loss and take profit based on ATR
                if result['prediction'] == 1:  # BUY
                    result['stop_loss'] = latest_price - (atr * SignalConfig.ATR_MULTIPLIER_SL)
                    result['take_profit'] = latest_price + (atr * SignalConfig.ATR_MULTIPLIER_TP)
                else:  # SELL
                    result['stop_loss'] = latest_price + (atr * SignalConfig.ATR_MULTIPLIER_SL)
                    result['take_profit'] = latest_price - (atr * SignalConfig.ATR_MULTIPLIER_TP)
            else:
                # Simple percentage-based stop loss and take profit
                if result['prediction'] == 1:  # BUY
                    result['stop_loss'] = latest_price * 0.98  # 2% stop loss
                    result['take_profit'] = latest_price * 1.04  # 4% take profit
                else:  # SELL
                    result['stop_loss'] = latest_price * 1.02  # 2% stop loss
                    result['take_profit'] = latest_price * 0.96  # 4% take profit
            
            # Calculate risk-reward ratio
            if result['prediction'] == 1:  # BUY
                risk = latest_price - result['stop_loss']
                reward = result['take_profit'] - latest_price
            else:  # SELL
                risk = result['stop_loss'] - latest_price
                reward = latest_price - result['take_profit']
            
            result['risk_reward'] = abs(reward / risk) if risk != 0 else 0
            result['entry_price'] = latest_price
            
            logger.info(f"Analysis for {symbol}: prediction={result['prediction']}, confidence={result['confidence']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {str(e)}")
            return None
    
    async def analyze_all_markets(self) -> Dict[str, Dict]:
        """
        Analyze all markets and generate predictions
        
        Returns:
            Dictionary mapping symbols to their prediction details
        """
        all_results = {}
        tasks = []
        
        # Get all symbols
        all_symbols = []
        for category, symbols in DataConfig.SYMBOLS.items():
            all_symbols.extend(symbols)
        
        # Create tasks for each symbol
        for symbol in all_symbols:
            if symbol in self.models:
                task = asyncio.create_task(self.analyze_market(symbol))
                tasks.append((symbol, task))
        
        # Wait for all tasks to complete
        for symbol, task in tasks:
            try:
                result = await task
                if result:
                    all_results[symbol] = result
            except Exception as e:
                logger.error(f"Error in task for {symbol}: {str(e)}")
        
        self.last_update = datetime.datetime.now()
        return all_results
    
    async def get_best_signal(self) -> Optional[Dict]:
        """
        Get the best trading signal across all markets
        
        Returns:
            Dictionary with the best signal details or None if no signals
        """
        # Get all market analyses
        analyses = await self.analyze_all_markets()
        
        if not analyses:
            return None
        
        # Filter signals by confidence threshold
        valid_signals = []
        
        for symbol, analysis in analyses.items():
            if analysis['confidence'] >= ModelConfig.CONFIDENCE_THRESHOLD:
                # Check if signal for this symbol was recently sent
                recent_signal = False
                cooldown_time = datetime.datetime.now() - datetime.timedelta(hours=SignalConfig.SIGNAL_COOLDOWN_HOURS)
                
                for signal in self.signals_history:
                    if signal['symbol'] == symbol and signal['timestamp'] > cooldown_time:
                        recent_signal = True
                        break
                
                if not recent_signal:
                    # Check risk-reward ratio
                    if analysis['risk_reward'] >= SignalConfig.MIN_RISK_REWARD_RATIO:
                        valid_signals.append(analysis)
        
        if not valid_signals:
            return None
        
        # Sort by confidence * risk_reward to get the best signal
        valid_signals.sort(key=lambda x: x['confidence'] * x['risk_reward'], reverse=True)
        best_signal = valid_signals[0]
        
        # Add to signals history
        self.signals_history.append(best_signal)
        
        # Keep only recent signals in history
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=7)
        self.signals_history = [s for s in self.signals_history if s['timestamp'] > cutoff_time]
        
        return best_signal
    
    def get_status(self) -> Dict:
        """
        Get the current status of the signal generator
        
        Returns:
            Dictionary with status information
        """
        # Count signals generated today
        today = datetime.datetime.now().date()
        signals_today = sum(1 for s in self.signals_history if s['timestamp'].date() == today)
        
        # Count available symbols with models
        symbol_count = len(self.models)
        
        # Calculate performance metrics if available
        performance = {}
        
        return {
            'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'last_update': self.last_update.strftime("%Y-%m-%d %H:%M:%S") if self.last_update else "Never",
            'symbol_count': symbol_count,
            'signals_today': signals_today,
            'performance': performance
        }
