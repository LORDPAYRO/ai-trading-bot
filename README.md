# AI Trading Assistant Bot

A Telegram-based trading bot that uses deep learning to analyze cryptocurrency and forex markets, detect profitable trading opportunities, and send high-confidence signals to users.

## Features

- **Multi-market Analysis**: Monitors multiple cryptocurrency pairs (BTC/USDT, ETH/USDT, etc.) and forex pairs (EUR/USD, USD/JPY, etc.)
- **AI-Powered Predictions**: Uses LSTM neural networks to predict price movements with confidence scores
- **Signal Generation**: Automatically calculates entry points, stop-loss, and take-profit levels
- **Telegram Integration**: Sends trading signals directly to users via Telegram bot
- **Risk Management**: Filters signals based on risk-reward ratio and confidence thresholds
- **Pocket Option Integration**: Direct WebSocket connection to Pocket Option for real-time price data

## Project Structure

```
AI_Trading_Bot/
├── data/                  # Data collection and preprocessing
│   ├── fetch_data.py      # Data fetching from exchanges
│   └── preprocess.py      # Data normalization and preparation
├── model/                 # ML models and training
│   ├── lstm_model.py      # LSTM model implementation
│   └── train_model.py     # Model training procedures
├── bot/                   # Bot implementation
│   ├── signal_logic.py    # Signal generation logic
│   └── telegram_bot.py    # Telegram bot interface
├── utils/                 # Utilities
│   ├── config.py          # Configuration settings
│   └── logger.py          # Logging setup
├── saved_models/          # Saved trained models
├── logs/                  # Application logs
├── plots/                 # Training history plots
├── pocket_ws.py           # Pocket Option WebSocket client
├── requirements.txt       # Dependencies
├── main.py                # Main application entry point
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AI_Trading_Bot.git
   cd AI_Trading_Bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   export TELEGRAM_API_TOKEN="your_telegram_bot_token"
   export BINANCE_API_KEY="your_binance_api_key"
   export BINANCE_API_SECRET="your_binance_api_secret"
   export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_api_key"
   export POCKET_OPTION_API_TOKEN="your_pocket_option_api_token"
   ```

   For Windows:
   ```
   set TELEGRAM_API_TOKEN=your_telegram_bot_token
   set BINANCE_API_KEY=your_binance_api_key
   set BINANCE_API_SECRET=your_binance_api_secret
   set ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   set POCKET_OPTION_API_TOKEN=your_pocket_option_api_token
   ```

## Usage

### Check Data Availability

To verify data can be fetched correctly:

```
python main.py --check-data
```

### Train Models

To train models for all configured symbols:

```
python main.py --train
```

### Run the Bot

To start the Telegram bot:

```
python main.py
```

### Pocket Option WebSocket

To run only the Pocket Option WebSocket client:

```
python main.py --pocket-ws
```

### Combined Mode

To run both the Telegram bot and Pocket Option WebSocket client:

```
python main.py --combined
```

## Telegram Bot Commands

- `/start` - Start the bot
- `/help` - Show available commands
- `/analyze` - Get current market analysis
- `/signal` - Get the strongest trading signal
- `/status` - Check bot status and performance
- `/list` - List all monitored symbols

## Configuration

You can customize the bot by editing the settings in `utils/config.py`:

- **Symbols**: Add or remove cryptocurrency and forex pairs
- **Timeframes**: Change the time intervals for analysis
- **Model Parameters**: Adjust LSTM architecture, window size, etc.
- **Signal Configuration**: Modify confidence thresholds, risk-reward ratio, etc.
- **Telegram Settings**: Set authorized users and chat IDs

## Pocket Option Integration

The Pocket Option WebSocket client (`pocket_ws.py`) provides:

- Real-time price data streaming for trading pairs
- Automatic reconnection with exponential backoff
- Authentication using API token
- Persistent subscription management
- Structured message handling for easy AI model integration

To integrate with the AI model, look for the following TODOs in `pocket_ws.py`:

```python
# TODO: Integrate with AI model for analysis
# Example: await self.ai_model.analyze_price(symbol, price, timestamp)

# TODO: Generate trading signals based on AI analysis
# Example: signal = await self.signal_generator.evaluate(symbol, price, analysis)
```

## Adding New Symbols

To add new symbols for monitoring:

1. Edit `utils/config.py` and add the symbol to the appropriate category in `DataConfig.SYMBOLS`
2. Run model training for the new symbol:
   ```
   python -c "from model.train_model import ModelTrainer; trainer = ModelTrainer(); trainer.train_model_for_symbol('NEW/SYMBOL')"
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies and forex involves significant risk of loss. Do not invest money you cannot afford to lose. Always do your own research before making any trading decisions.
