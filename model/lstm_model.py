import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import joblib

from utils.config import ModelConfig
from utils.logger import logger

class LSTMModel:
    """LSTM-based model for time series forecasting"""
    
    def __init__(self, input_shape=None):
        """
        Initialize the LSTM model
        
        Args:
            input_shape: Shape of input data (window_size, features)
        """
        self.model = None
        self.input_shape = input_shape
        self.history = None
    
    def build_model(self, input_shape=None):
        """
        Build the LSTM model architecture
        
        Args:
            input_shape: Shape of input data (window_size, features)
            
        Returns:
            Compiled Keras model
        """
        if input_shape:
            self.input_shape = input_shape
        
        if not self.input_shape:
            raise ValueError("Input shape must be provided")
        
        # Create Sequential model
        model = Sequential([
            # LSTM layer
            LSTM(units=ModelConfig.LSTM_UNITS, 
                 return_sequences=True, 
                 input_shape=self.input_shape),
            
            # Dropout for regularization
            Dropout(ModelConfig.DROPOUT_RATE),
            
            # Second LSTM layer
            LSTM(units=ModelConfig.LSTM_UNITS, 
                 return_sequences=False),
            
            # Dropout for regularization
            Dropout(ModelConfig.DROPOUT_RATE),
            
            # Dense hidden layer
            Dense(units=ModelConfig.DENSE_UNITS, activation='relu'),
            
            # Dropout for regularization
            Dropout(ModelConfig.DROPOUT_RATE),
            
            # Output layer - binary classification (up/down)
            Dense(units=1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=ModelConfig.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Built LSTM model with input shape {self.input_shape}")
        
        return model
    
    def train(self, X_train, y_train, validation_data=None):
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            validation_data: Tuple of (X_val, y_val) for validation
            
        Returns:
            Training history
        """
        if self.model is None:
            # Infer input shape from training data
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Create model directory if it doesn't exist
        os.makedirs(ModelConfig.MODEL_SAVE_PATH, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=os.path.join(ModelConfig.MODEL_SAVE_PATH, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=ModelConfig.EPOCHS,
            batch_size=ModelConfig.BATCH_SIZE,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        logger.info(f"Model trained for {len(history.epoch)} epochs")
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        return self.model.predict(X)
    
    def predict_with_confidence(self, X):
        """
        Make predictions with confidence scores
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidences)
            - predictions: Binary predictions (0 or 1)
            - confidences: Confidence scores (0.0 to 1.0)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Get raw probabilities
        probabilities = self.model.predict(X)
        
        # Convert to binary predictions
        predictions = (probabilities > 0.5).astype(int)
        
        # Calculate confidence scores (distance from 0.5)
        confidences = np.abs(probabilities - 0.5) * 2
        
        return predictions, confidences
    
    def save(self, filepath=None):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            filepath = os.path.join(ModelConfig.MODEL_SAVE_PATH, 'lstm_model.h5')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        
        # Save training history if available
        if self.history is not None:
            history_path = os.path.join(os.path.dirname(filepath), 'training_history.pkl')
            joblib.dump(self.history, history_path)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath=None):
        """
        Load the model from disk
        
        Args:
            filepath: Path to load the model from
        """
        if filepath is None:
            filepath = os.path.join(ModelConfig.MODEL_SAVE_PATH, 'lstm_model.h5')
        
        if not os.path.exists(filepath):
            raise ValueError(f"Model file not found: {filepath}")
        
        # Load the model
        self.model = load_model(filepath)
        
        # Load training history if available
        history_path = os.path.join(os.path.dirname(filepath), 'training_history.pkl')
        if os.path.exists(history_path):
            self.history = joblib.load(history_path)
        
        # Update input shape
        self.input_shape = self.model.layers[0].input_shape[1:]
        
        logger.info(f"Model loaded from {filepath}")
        
        return self.model
