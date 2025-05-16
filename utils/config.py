import os
from typing import Dict, List, Union, Optional

# API Keys and Authentication
TELEGRAM_API_TOKEN = os.environ.get("TELEGRAM_API_TOKEN", "YOUR_TELEGRAM_TOKEN")
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

# Data Collection Settings
class DataConfig:
    SYMBOLS = {
        "crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "BNB/USDT"],
        "forex": ["EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD", "USD/CAD"]
    }
    
    DATA_SOURCES = {
        "crypto": "binance",
        "forex": "alpha_vantage"
    }
    
    TIMEFRAMES = ["15m", "1h", "4h", "1d"]
    DEFAULT_TIMEFRAME = "1h"
    
    # Historical data parameters
    LOOKBACK_PERIOD_DAYS = 180  # 6 months of historical data
    
    # Real-time data parameters
    UPDATE_INTERVAL_SECONDS = 300  # Update every 5 minutes

# Model Settings
class ModelConfig:
    # LSTM Model Parameters
    WINDOW_SIZE = 60  # Past time steps to consider
    FEATURES = ["open", "high", "low", "close", "volume"]
    OUTPUT_SIZE = 1  # Direction prediction (1 for up/down)
    
    # LSTM Architecture
    LSTM_UNITS = 100
    DROPOUT_RATE = 0.2
    DENSE_UNITS = 64
    
    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001
    
    # Save/Load paths
    MODEL_SAVE_PATH = "saved_models/"
    
    # Prediction Settings
    CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence to generate signal

# Signal Generation Settings
class SignalConfig:
    # Stop-loss and take-profit calculation
    USE_ATR_MULTIPLIER = True  # Use Average True Range for SL/TP
    ATR_PERIOD = 14
    ATR_MULTIPLIER_TP = 3.0
    ATR_MULTIPLIER_SL = 1.5
    
    # Signal filtering
    MIN_RISK_REWARD_RATIO = 1.5
    
    # Cooldown between signals for the same symbol
    SIGNAL_COOLDOWN_HOURS = 4

# Telegram Bot Settings
class TelegramConfig:
    ADMIN_USER_IDS = [12345678]  # User IDs of admins
    AUTHORIZED_CHAT_IDS = [-1001234567890]  # Chat IDs of authorized groups/channels
    
    # Command list
    COMMANDS = {
        "start": "Start the bot",
        "analyze": "Get current market analysis",
        "signal": "Get the strongest trading signal",
        "status": "Check bot status and performance",
        "list": "List all monitored symbols"
    }
    
    # Message templates
    SIGNAL_MESSAGE_TEMPLATE = """
🔔 *TRADING SIGNAL ALERT* 🔔

Symbol: {symbol}
Direction: {direction}
Confidence: {confidence:.2f}%

Entry Price: {entry_price:.2f}
Take Profit: {take_profit:.2f}
Stop Loss: {stop_loss:.2f}

Timeframe: {timeframe}
Signal Time: {signal_time}

Risk/Reward Ratio: {risk_reward:.2f}
    """

# Logging Settings
class LogConfig:
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/trading_bot.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 5
