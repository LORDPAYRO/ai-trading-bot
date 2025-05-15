import asyncio
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set

import pandas as pd

from ai_hook.analyzer import MarketAnalyzer
from pocket_ws import PocketOptionWebSocket
from utils.logger import logger

class AIHookController:
    """
    Controller that integrates the market analyzer with data sources
    and the Telegram bot for sending signals.
    """
    
    def __init__(self, telegram_bot=None):
        """
        Initialize the AI Hook Controller
        
        Args:
            telegram_bot: Telegram bot instance for sending signals
        """
        self.analyzer = MarketAnalyzer()
        self.telegram_bot = telegram_bot
        self.data_sources = {}  # symbol -> data source mapping
        self.is_running = False
        self.scan_interval = 30  # seconds between market scans
        self.signal_cooldown = {}  # symbol -> last signal time mapping
        self.cooldown_period = 3600  # 1 hour cooldown between signals for same symbol
        self.signal_queue = asyncio.Queue()  # Queue for signals to be sent
        self.active_symbols = set()  # Set of symbols being monitored
        self.pending_pre_alerts = {}  # symbol -> pre-alert task mapping
        self.pre_alert_delay = 60  # Seconds between pre-alert and final signal (default: 60s)
        
        # Create directory for screenshots if we want to include them later
        os.makedirs("ai_hook/screenshots", exist_ok=True)
        
        logger.info("AI Hook Controller initialized")
    
    async def connect_to_pocket_option(self, symbols: List[str] = None):
        """
        Connect to Pocket Option WebSocket and start data streaming
        
        Args:
            symbols: List of symbols to monitor (e.g., ["BTC/USD"])
        """
        if symbols is None:
            symbols = ["BTC/USD"]  # Default symbol
        
        try:
            # Initialize WebSocket client
            api_token = os.environ.get("POCKET_OPTION_API_TOKEN")
            ws_client = PocketOptionWebSocket(api_token=api_token)
            
            # Store connection for later use
            self.pocket_ws = ws_client
            
            # Register data handler
            self._register_pocket_option_handler(ws_client)
            
            # Start connection in background
            self.pocket_connection_task = asyncio.create_task(ws_client.connect())
            
            # Wait for connection to establish
            await asyncio.sleep(2)
            
            # Subscribe to symbols
            for symbol in symbols:
                if ws_client.connected:
                    await ws_client.subscribe(symbol)
                    self.active_symbols.add(symbol)
            
            logger.info(f"Connected to Pocket Option and subscribed to: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Pocket Option: {str(e)}")
            return False
    
    def _register_pocket_option_handler(self, ws_client):
        """
        Register a custom message handler for Pocket Option data
        
        Args:
            ws_client: PocketOptionWebSocket instance
        """
        # Store the original handler
        original_handler = ws_client.handle_price_update
        
        # Create a new handler that forwards data to our analyzer
        async def custom_handler(price_data):
            # First, call the original handler
            await original_handler(price_data)
            
            # Extract data for our analyzer
            symbol = price_data.get("symbol", "unknown")
            price = price_data.get("price")
            timestamp = price_data.get("timestamp")
            
            if price is not None and symbol in self.active_symbols:
                # Convert to DataFrame
                if timestamp is None:
                    timestamp = datetime.now()
                
                # For simplicity, we're creating a basic OHLCV dataframe with just the current price
                # In a real implementation, you'd accumulate these into proper candles
                df = pd.DataFrame([{
                    'timestamp': timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0  # We may not have volume data
                }])
                df.set_index('timestamp', inplace=True)
                
                # Send to analyzer
                await self._process_new_data(symbol, df)
        
        # Replace the handler
        ws_client.handle_price_update = custom_handler
    
    async def _process_new_data(self, symbol: str, data: pd.DataFrame):
        """
        Process new market data through the analyzer
        
        Args:
            symbol: Symbol identifier
            data: DataFrame with market data
        """
        try:
            # Process the data through the analyzer
            result = await self.analyzer.process_market_data(symbol, data)
            
            # If we have a signal with sufficient confidence, add it to the queue
            if result.get("signal") is not None:
                # Check if we're in cooldown period for this symbol
                if symbol in self.signal_cooldown:
                    last_signal_time = self.signal_cooldown[symbol]
                    if datetime.now() - last_signal_time < timedelta(seconds=self.cooldown_period):
                        logger.info(f"Signal for {symbol} is in cooldown period. Skipping.")
                        return
                
                # Check if we already have a pending pre-alert for this symbol
                if symbol in self.pending_pre_alerts:
                    logger.info(f"Pre-alert for {symbol} already pending. Skipping duplicate signal.")
                    return
                
                # Generate a unique signal ID
                signal_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                result["signal_id"] = signal_id
                
                # Store current price for later comparison
                entry_price = data.iloc[-1]["close"]
                result["entry_price"] = entry_price
                
                # Schedule a pre-alert followed by the actual signal
                self.pending_pre_alerts[symbol] = asyncio.create_task(
                    self._handle_signal_with_pre_alert(symbol, result)
                )
                
                logger.info(f"Scheduled pre-alert and signal for: {symbol} - {result['signal']} (Confidence: {result['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
    
    async def _handle_signal_with_pre_alert(self, symbol: str, signal: Dict[str, Any]):
        """
        Handle a signal by sending pre-alert first, then the actual signal
        
        Args:
            symbol: Symbol identifier
            signal: Signal data
        """
        try:
            # Send pre-alert
            await self._send_pre_alert(signal)
            
            # Wait for the configured delay
            await asyncio.sleep(self.pre_alert_delay)
            
            # Add actual signal to queue
            await self.signal_queue.put(signal)
            
            # Update cooldown
            self.signal_cooldown[symbol] = datetime.now()
            
            # Clean up pending pre-alert
            self.pending_pre_alerts.pop(symbol, None)
            
        except asyncio.CancelledError:
            logger.info(f"Signal handling for {symbol} was cancelled")
            self.pending_pre_alerts.pop(symbol, None)
            
        except Exception as e:
            logger.error(f"Error in signal handling for {symbol}: {str(e)}")
            self.pending_pre_alerts.pop(symbol, None)
    
    async def _send_pre_alert(self, signal: Dict[str, Any]):
        """
        Send a pre-alert for an upcoming signal
        
        Args:
            signal: Signal data
        """
        try:
            # Format the pre-alert message
            minutes = self.pre_alert_delay // 60
            seconds = self.pre_alert_delay % 60
            time_format = f"{minutes:02d}:{seconds:02d}" if minutes > 0 else f"00:{seconds:02d}"
            
            direction_emoji = "⬆️⬆️⬆️" if signal["signal"] == "UP" else "⬇️⬇️⬇️"
            
            message = f"""
❗️Set timer to {time_format}❗️

{signal['symbol']} {signal.get('market_type', '')}
{signal['signal']} {direction_emoji}
1 MIN
"""
            
            # Send to all authorized chats
            from utils.config import TelegramConfig
            for chat_id in TelegramConfig.AUTHORIZED_CHAT_IDS:
                await self.telegram_bot.application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="Markdown"
                )
            
            logger.info(f"Pre-alert sent for {signal['symbol']}")
            
        except Exception as e:
            logger.error(f"Error sending pre-alert: {str(e)}")
    
    async def start_signal_sender(self):
        """Start the background task that sends signals from the queue"""
        if not self.telegram_bot:
            logger.warning("Telegram bot not provided. Signals will be logged but not sent.")
        
        self.signal_task = asyncio.create_task(self._signal_sender_loop())
        logger.info("Signal sender started")
    
    async def _signal_sender_loop(self):
        """Background loop that sends signals from the queue"""
        while True:
            try:
                # Get the next signal from the queue
                signal = await self.signal_queue.get()
                
                # Log the signal
                logger.info(f"Processing signal: {signal['symbol']} - {signal['signal']} ({signal['confidence']:.2f})")
                
                # Send the signal via Telegram if bot is available
                if self.telegram_bot:
                    await self._send_telegram_signal(signal)
                
                # Mark task as done
                self.signal_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Signal sender loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in signal sender loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before continuing
    
    async def _send_telegram_signal(self, signal):
        """
        Send a trading signal to Telegram
        
        Args:
            signal: Signal data to send
        """
        try:
            # Format the signal message
            message = self._format_signal_message(signal)
            
            # Send to all authorized chats
            from utils.config import TelegramConfig
            for chat_id in TelegramConfig.AUTHORIZED_CHAT_IDS:
                await self.telegram_bot.application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="Markdown"
                )
            
            logger.info(f"Signal sent to Telegram for {signal['symbol']}")
            
            # Save the signal ID for later validation
            signal_id = signal.get('signal_id', f"{signal['symbol']}_{int(time.time())}")
            
            # TODO: Schedule a validation check after some time
            # This would check if the prediction was correct
            
        except Exception as e:
            logger.error(f"Error sending signal to Telegram: {str(e)}")
    
    def _format_signal_message(self, signal):
        """
        Format a signal for Telegram message
        
        Args:
            signal: Signal data
            
        Returns:
            Formatted message string
        """
        # Get emoji based on direction
        direction_emoji = "🔼" if signal["signal"] == "UP" else "🔽"
        
        # Format message
        message = f"""
🔔 New Signal!

Pair: {signal['symbol']}
Direction: {direction_emoji} {signal['signal']}
Confidence: {signal['confidence']*100:.1f}%
Reason: {signal['reason']}
Signal ID: {signal.get('signal_id', 'Unknown')}
"""
        return message
    
    async def start(self, symbols=None):
        """
        Start the AI Hook Controller
        
        Args:
            symbols: List of symbols to monitor
        """
        if self.is_running:
            logger.warning("AI Hook Controller is already running")
            return
        
        self.is_running = True
        
        # Connect to Pocket Option
        connected = await self.connect_to_pocket_option(symbols)
        if not connected:
            logger.error("Failed to start AI Hook: Could not connect to data source")
            self.is_running = False
            return False
        
        # Start signal sender
        await self.start_signal_sender()
        
        logger.info("AI Hook Controller started successfully")
        return True
    
    async def stop(self):
        """Stop the AI Hook Controller"""
        if not self.is_running:
            return
        
        # Cancel pending pre-alerts
        for symbol, task in list(self.pending_pre_alerts.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.pending_pre_alerts.clear()
        
        # Cancel tasks
        if hasattr(self, 'signal_task'):
            self.signal_task.cancel()
            try:
                await self.signal_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, 'pocket_connection_task'):
            self.pocket_connection_task.cancel()
            try:
                await self.pocket_connection_task
            except asyncio.CancelledError:
                pass
        
        self.is_running = False
        logger.info("AI Hook Controller stopped")
    
    async def get_stats(self):
        """
        Get statistics about the AI Hook performance
        
        Returns:
            Dictionary with statistics
        """
        # Get accuracy stats from analyzer
        accuracy_stats = self.analyzer.get_accuracy_stats()
        
        # Add additional stats
        stats = {
            "accuracy": accuracy_stats,
            "active_symbols": list(self.active_symbols),
            "signals_in_queue": self.signal_queue.qsize(),
            "pending_pre_alerts": len(self.pending_pre_alerts),
            "pre_alert_delay": self.pre_alert_delay,
            "uptime": "N/A"  # Would calculate from start time
        }
        
        return stats
    
    async def add_symbol(self, symbol):
        """
        Add a new symbol to monitor
        
        Args:
            symbol: Symbol to add
            
        Returns:
            True if successful, False otherwise
        """
        if symbol in self.active_symbols:
            logger.warning(f"Symbol {symbol} is already being monitored")
            return False
        
        try:
            # Subscribe to the symbol
            if hasattr(self, 'pocket_ws') and self.pocket_ws.connected:
                await self.pocket_ws.subscribe(symbol)
                self.active_symbols.add(symbol)
                logger.info(f"Added {symbol} to monitored symbols")
                return True
            else:
                logger.error(f"Cannot add symbol {symbol}: WebSocket not connected")
                return False
                
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {str(e)}")
            return False
    
    async def remove_symbol(self, symbol):
        """
        Remove a symbol from monitoring
        
        Args:
            symbol: Symbol to remove
            
        Returns:
            True if successful, False otherwise
        """
        if symbol not in self.active_symbols:
            logger.warning(f"Symbol {symbol} is not being monitored")
            return False
        
        try:
            # Cancel any pending pre-alerts for this symbol
            if symbol in self.pending_pre_alerts:
                task = self.pending_pre_alerts.pop(symbol)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Unsubscribe from the symbol
            if hasattr(self, 'pocket_ws') and self.pocket_ws.connected:
                await self.pocket_ws.unsubscribe(symbol)
                self.active_symbols.remove(symbol)
                logger.info(f"Removed {symbol} from monitored symbols")
                return True
            else:
                logger.error(f"Cannot remove symbol {symbol}: WebSocket not connected")
                return False
                
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {str(e)}")
            return False
    
    def set_scan_interval(self, seconds):
        """
        Set the interval between market scans
        
        Args:
            seconds: Interval in seconds
        """
        if seconds < 5:
            logger.warning("Scan interval too short. Setting to minimum of 5 seconds.")
            seconds = 5
        
        self.scan_interval = seconds
        logger.info(f"Scan interval set to {seconds} seconds")
    
    def set_signal_confidence_threshold(self, threshold):
        """
        Set the minimum confidence threshold for sending signals
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if threshold < 0 or threshold > 1:
            logger.warning("Confidence threshold must be between 0 and 1. Using default.")
            return
        
        self.analyzer.min_confidence = threshold
        logger.info(f"Signal confidence threshold set to {threshold}")
    
    def set_cooldown_period(self, hours):
        """
        Set the cooldown period between signals for the same symbol
        
        Args:
            hours: Cooldown period in hours
        """
        if hours < 0:
            logger.warning("Cooldown period cannot be negative. Using default.")
            return
        
        self.cooldown_period = hours * 3600  # Convert hours to seconds
        logger.info(f"Signal cooldown period set to {hours} hours")
    
    def set_pre_alert_delay(self, seconds):
        """
        Set the delay between pre-alert and final signal
        
        Args:
            seconds: Delay in seconds
        """
        if seconds < 10:
            logger.warning("Pre-alert delay too short. Setting to minimum of 10 seconds.")
            seconds = 10
        
        self.pre_alert_delay = seconds
        logger.info(f"Pre-alert delay set to {seconds} seconds") 