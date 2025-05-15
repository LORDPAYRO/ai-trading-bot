import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pocket_ws.log")
    ]
)
logger = logging.getLogger("pocket_ws")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

class PocketOptionWebSocket:
    """
    WebSocket client for Pocket Option that handles connections, authentication,
    subscription to trading pairs, and reconnection logic.
    """
    
    def __init__(self, uri: str = "wss://api.pocketoption.com", api_token: Optional[str] = None):
        """
        Initialize the Pocket Option WebSocket client.
        
        Args:
            uri: WebSocket URI for Pocket Option API
            api_token: API token for authentication (optional)
        """
        self.uri = uri
        self.api_token = api_token or os.environ.get("POCKET_OPTION_API_TOKEN")
        self.websocket = None
        self.connected = False
        self.subscriptions: Set[str] = set()  # Track active subscriptions
        self.reconnect_interval = 1  # Initial reconnect delay in seconds
        self.max_reconnect_interval = 60  # Maximum reconnect delay
        self.last_message_time = 0  # Track last message timestamp for heartbeat
        self.heartbeat_interval = 30  # Send heartbeat every 30 seconds
        
        # Statistics
        self.connect_count = 0
        self.message_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.print_stats()
        sys.exit(0)
    
    async def connect(self):
        """
        Establish WebSocket connection to Pocket Option with automatic reconnection.
        """
        while True:
            try:
                if self.websocket and not self.websocket.closed:
                    await self.websocket.close()
                
                logger.info(f"Connecting to {self.uri}...")
                self.websocket = await websockets.connect(self.uri)
                self.connected = True
                self.connect_count += 1
                self.reconnect_interval = 1  # Reset reconnect interval on successful connection
                logger.info("WebSocket connection established")
                
                # Authenticate if token is available
                if self.api_token:
                    await self.authenticate()
                
                # Resubscribe to previous subscriptions
                await self.resubscribe()
                
                # Process messages
                await self.message_loop()
                
            except (ConnectionClosed, ConnectionClosedError, ConnectionRefusedError) as e:
                self.connected = False
                self.error_count += 1
                logger.error(f"WebSocket connection error: {e}")
                
                # Implement exponential backoff for reconnection
                logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
                await asyncio.sleep(self.reconnect_interval)
                self.reconnect_interval = min(self.reconnect_interval * 2, self.max_reconnect_interval)
                
            except Exception as e:
                self.connected = False
                self.error_count += 1
                logger.error(f"Unexpected error: {str(e)}")
                await asyncio.sleep(self.reconnect_interval)
    
    async def authenticate(self):
        """
        Authenticate with Pocket Option API using the provided token.
        """
        if not self.api_token:
            logger.warning("No API token provided, skipping authentication")
            return
        
        try:
            auth_message = {
                "action": "auth",
                "token": self.api_token
            }
            await self.send_message(auth_message)
            logger.info("Authentication request sent")
            
            # Note: Actual implementation might need to wait for an auth response
            # This is a placeholder based on general WebSocket patterns
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise
    
    async def subscribe(self, symbol: str):
        """
        Subscribe to a trading pair to receive real-time price data.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
        """
        try:
            # Format symbol according to Pocket Option requirements
            # Note: Adjust the format based on actual API documentation
            formatted_symbol = symbol.replace("/", "")
            
            subscription_message = {
                "action": "subscribe",
                "pairs": [formatted_symbol]
            }
            
            await self.send_message(subscription_message)
            self.subscriptions.add(symbol)
            logger.info(f"Subscribed to {symbol}")
            
        except Exception as e:
            logger.error(f"Subscription error for {symbol}: {str(e)}")
            raise
    
    async def unsubscribe(self, symbol: str):
        """
        Unsubscribe from a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
        """
        try:
            # Format symbol according to Pocket Option requirements
            formatted_symbol = symbol.replace("/", "")
            
            unsubscribe_message = {
                "action": "unsubscribe",
                "pairs": [formatted_symbol]
            }
            
            await self.send_message(unsubscribe_message)
            self.subscriptions.discard(symbol)
            logger.info(f"Unsubscribed from {symbol}")
            
        except Exception as e:
            logger.error(f"Unsubscription error for {symbol}: {str(e)}")
    
    async def resubscribe(self):
        """
        Resubscribe to all previously subscribed pairs after reconnection.
        """
        if not self.subscriptions:
            return
        
        logger.info(f"Resubscribing to {len(self.subscriptions)} pairs...")
        for symbol in list(self.subscriptions):
            await self.subscribe(symbol)
    
    async def send_message(self, message: Dict):
        """
        Send a message to the WebSocket server.
        
        Args:
            message: Message to send as a dictionary
        """
        if not self.websocket or not self.connected:
            raise ConnectionError("WebSocket is not connected")
        
        message_str = json.dumps(message)
        await self.websocket.send(message_str)
    
    async def send_heartbeat(self):
        """
        Send a heartbeat message to keep the connection alive.
        """
        heartbeat_message = {
            "action": "ping",
            "timestamp": int(time.time())
        }
        await self.send_message(heartbeat_message)
        logger.debug("Heartbeat sent")
    
    async def handle_message(self, message_data: Dict[str, Any]):
        """
        Process incoming WebSocket messages.
        
        Args:
            message_data: Parsed JSON message from the WebSocket
        """
        # Update last message time
        self.last_message_time = time.time()
        self.message_count += 1
        
        # Log the message for debugging
        logger.debug(f"Received message: {message_data}")
        
        # Handle different message types
        message_type = message_data.get("type")
        
        if message_type == "price":
            # Extract and process price data
            await self.handle_price_update(message_data)
        elif message_type == "pong":
            # Heartbeat response
            logger.debug("Received heartbeat response")
        elif message_type == "auth":
            # Authentication response
            status = message_data.get("status")
            if status == "success":
                logger.info("Authentication successful")
            else:
                logger.error(f"Authentication failed: {message_data.get('message', 'Unknown error')}")
        else:
            # Handle other message types
            logger.info(f"Received message type: {message_type}")
    
    async def handle_price_update(self, price_data: Dict[str, Any]):
        """
        Process price update messages from the WebSocket.
        
        Args:
            price_data: Price update data
        """
        # Extract relevant data fields
        # Note: Adjust field names based on actual API response format
        symbol = price_data.get("symbol", "unknown")
        price = price_data.get("price")
        timestamp = price_data.get("timestamp")
        
        if price is not None:
            logger.info(f"Price update for {symbol}: {price} at {timestamp}")
            
            # TODO: Integrate with AI model for analysis
            # This is where you would pass the price data to your AI model
            # Example: await self.ai_model.analyze_price(symbol, price, timestamp)
            
            # TODO: Generate trading signals based on AI analysis
            # Example: signal = await self.signal_generator.evaluate(symbol, price, analysis)
            
            # Store data for later analysis (optional)
            # Example: self.data_store.add_price_point(symbol, price, timestamp)
        else:
            logger.warning(f"Received price update with missing data: {price_data}")
    
    async def heartbeat_loop(self):
        """
        Maintain a heartbeat to keep the connection alive.
        """
        while self.connected:
            current_time = time.time()
            if current_time - self.last_message_time > self.heartbeat_interval:
                try:
                    await self.send_heartbeat()
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {str(e)}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def message_loop(self):
        """
        Main loop to process incoming WebSocket messages.
        """
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        
        try:
            async for message in self.websocket:
                try:
                    # Parse the message (assuming JSON format)
                    message_data = json.loads(message)
                    
                    # Process the message
                    await self.handle_message(message_data)
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
        
        finally:
            # Cancel heartbeat task on exit
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    def print_stats(self):
        """
        Print connection statistics.
        """
        uptime = datetime.now() - self.start_time
        logger.info(f"WebSocket Client Statistics:")
        logger.info(f"Uptime: {uptime}")
        logger.info(f"Total connections: {self.connect_count}")
        logger.info(f"Messages received: {self.message_count}")
        logger.info(f"Errors encountered: {self.error_count}")
        logger.info(f"Active subscriptions: {len(self.subscriptions)}")

async def main():
    """
    Main entry point for the script.
    """
    # Load configuration (could be enhanced to load from a config file)
    api_token = os.environ.get("POCKET_OPTION_API_TOKEN")
    symbols_to_monitor = ["BTC/USD"]  # Add more symbols as needed
    
    # Create and connect WebSocket client
    client = PocketOptionWebSocket(api_token=api_token)
    
    # Create connection task
    connection_task = asyncio.create_task(client.connect())
    
    # Wait a moment for connection to establish
    await asyncio.sleep(2)
    
    # Subscribe to symbols
    for symbol in symbols_to_monitor:
        if client.connected:
            await client.subscribe(symbol)
    
    try:
        # Run indefinitely
        await connection_task
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    finally:
        # Print statistics on exit
        client.print_stats()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        raise 