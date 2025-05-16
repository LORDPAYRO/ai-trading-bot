import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

from data.fetch_data import DataManager
from data.preprocess import DataPreprocessor
from model.lstm_model import LSTMModel
from utils.config import DataConfig, ModelConfig
from utils.logger import logger

class ModelTrainer:
    """Handles model training for all symbols"""
    
    def __init__(self):
        """Initialize the model trainer"""
        self.data_manager = DataManager()
        self.preprocessor = DataPreprocessor()
        self.models = {}  # Dictionary to store models for each symbol
        logger.info("Initialized Model Trainer")
    
    def train_model_for_symbol(self, symbol, timeframe=None, save_model=True):
        """
        Train a model for a specific symbol
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval for data
            save_model: Whether to save the model after training
            
        Returns:
            Trained model and evaluation metrics
        """
        timeframe = timeframe or DataConfig.DEFAULT_TIMEFRAME
        
        try:
            # Fetch historical data
            logger.info(f"Fetching historical data for {symbol}...")
            df = self.data_manager.fetch_historical_data(symbol, timeframe)
            
            if df.empty:
                logger.error(f"No data available for {symbol}")
                return None, {}
            
            # Preprocess data
            logger.info(f"Preprocessing data for {symbol}...")
            X, y = self.preprocessor.prepare_data(df, symbol, is_training=True)
            
            if len(X) == 0 or len(y) == 0:
                logger.error(f"Failed to preprocess data for {symbol}")
                return None, {}
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=ModelConfig.VALIDATION_SPLIT,
                shuffle=False  # Time series data should not be shuffled
            )
            
            logger.info(f"Training data shape for {symbol}: {X_train.shape}")
            logger.info(f"Validation data shape for {symbol}: {X_val.shape}")
            
            # Create and build model
            model = LSTMModel()
            input_shape = (X_train.shape[1], X_train.shape[2])
            model.build_model(input_shape)
            
            # Train the model
            logger.info(f"Training model for {symbol}...")
            history = model.train(X_train, y_train, validation_data=(X_val, y_val))
            
            # Evaluate the model
            logger.info(f"Evaluating model for {symbol}...")
            evaluation = model.model.evaluate(X_val, y_val, verbose=0)
            metrics = {
                'val_loss': evaluation[0],
                'val_accuracy': evaluation[1]
            }
            
            logger.info(f"Model for {symbol} achieved validation accuracy: {metrics['val_accuracy']:.4f}")
            
            # Save the model
            if save_model:
                os.makedirs(ModelConfig.MODEL_SAVE_PATH, exist_ok=True)
                model_path = os.path.join(ModelConfig.MODEL_SAVE_PATH, f"model_{symbol.replace('/', '_')}.h5")
                model.save(model_path)
                
                # Save preprocessor scalers
                self.preprocessor.save_scalers()
            
            # Store the model
            self.models[symbol] = model
            
            # Plot and save training history
            self._plot_training_history(model.history, symbol)
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return None, {}
    
    def train_all_models(self, timeframe=None, symbols=None):
        """
        Train models for all symbols or a subset of symbols
        
        Args:
            timeframe: Time interval for data
            symbols: List of symbols to train models for (if None, use all symbols)
            
        Returns:
            Dictionary mapping symbols to their metrics
        """
        all_metrics = {}
        
        # Get symbols to train
        if symbols is None:
            symbols = []
            for category, symbol_list in DataConfig.SYMBOLS.items():
                symbols.extend(symbol_list)
        
        for symbol in symbols:
            logger.info(f"Training model for symbol: {symbol}")
            _, metrics = self.train_model_for_symbol(symbol, timeframe)
            all_metrics[symbol] = metrics
        
        return all_metrics
    
    def load_models(self):
        """
        Load all saved models from disk
        
        Returns:
            Dictionary mapping symbols to loaded models
        """
        self.models = {}
        
        if not os.path.exists(ModelConfig.MODEL_SAVE_PATH):
            logger.warning(f"Model directory {ModelConfig.MODEL_SAVE_PATH} does not exist")
            return self.models
        
        # Load preprocessor scalers
        self.preprocessor.load_scalers()
        
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
        return self.models
    
    def _plot_training_history(self, history, symbol):
        """
        Plot and save training history
        
        Args:
            history: Training history dictionary
            symbol: Symbol for the model
        """
        if not history:
            return
        
        os.makedirs("plots", exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{symbol} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{symbol} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        filename = f"plots/training_history_{symbol.replace('/', '_')}.png"
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Saved training history plot to {filename}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Train models for all symbols
    trainer.train_all_models()
    
    # Or train for a specific symbol
    # trainer.train_model_for_symbol("BTC/USDT")
