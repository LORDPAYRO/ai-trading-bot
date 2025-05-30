# -*- coding: utf-8 -*-
import os
import argparse
import asyncio
import logging
from datetime import datetime

from data.fetch_data import DataManager
from model.train_model import ModelTrainer
from bot.telegram_bot import TelegramBot
from utils.logger import logger
from utils.config import ModelConfig
from pocket_ws import PocketOptionWebSocket

def setup_environment():
    """Set up the environment by creating necessary directories"""
    os.makedirs(ModelConfig.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("ai_hook/models", exist_ok=True)
    os.makedirs("ai_hook/screenshots", exist_ok=True)
    logger.info("Environment setup complete")

def check_api_keys():
    """Check if the required API keys are set"""
    from utils.config import TELEGRAM_API_TOKEN, BINANCE_API_KEY, BINANCE_API_SECRET
    missing_keys = []
    if not TELEGRAM_API_TOKEN or TELEGRAM_API_TOKEN == "YOUR_TELEGRAM_TOKEN":
        missing_keys.append("TELEGRAM_API_TOKEN")
    if not BINANCE_API_KEY:
        missing_keys.append("BINANCE_API_KEY")
    if not BINANCE_API_SECRET:
        missing_keys.append("BINANCE_API_SECRET")
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        logger.warning("Please set these environment variables before running the bot")
        return False
    return True

def train_models():
    """Train models for all configured symbols"""
    trainer = ModelTrainer()
    trainer.train_all_models()

def run_bot():
    """Run the Telegram bot"""
    bot = TelegramBot()
    bot.run()

async def run_pocket_ws(symbols=None):
    """Run the Pocket Option WebSocket client"""
    if symbols is None:
        symbols = ["BTC/USD"]
    api_token = os.environ.get("POCKET_OPTION_API_TOKEN")
    client = PocketOptionWebSocket(api_token=api_token)
    connection_task = asyncio.create_task(client.connect())
    await asyncio.sleep(2)
    for symbol in symbols:
        if client.connected:
            await client.subscribe(symbol)
    try:
        await connection_task
    except asyncio.CancelledError:
        logger.info("Pocket Option WebSocket task cancelled")
    finally:
        client.print_stats()

async def run_combined():
    """Run both Telegram bot and Pocket Option WebSocket client"""
    ws_task = asyncio.create_task(run_pocket_ws())
    import threading
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        logger.info("Combined task cancelled")
    finally:
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass

async def run_ai_bot():
    """Run the Telegram bot with AI Hook integration"""
    from ai_hook.controller import AIHookController
    bot = TelegramBot()
    import threading
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()
    await asyncio.sleep(2)
    logger.info("AI Hook integration ready. Use /ai_start in Telegram to begin monitoring.")
    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        logger.info("AI bot task cancelled")
    except KeyboardInterrupt:
        logger.info("AI bot interrupted by user")

def main():
    """Main function to run the trading bot"""
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--train", action="store_true", help="Train models before starting the bot")
    parser.add_argument("--check-data", action="store_true", help="Check data availability without training models")
    parser.add_argument("--pocket-ws", action="store_true", help="Run only the Pocket Option WebSocket client")
    parser.add_argument("--combined", action="store_true", help="Run both Telegram bot and Pocket Option WebSocket")
    parser.add_argument("--ai", action="store_true", help="Run the bot with AI Hook integration")
    args = parser.parse_args()

    setup_environment()

    keys_ok = check_api_keys()
    if not keys_ok:
        logger.error("Missing required API keys. Exiting.")
        return

    if args.check_data:
        logger.info("Checking data availability...")
        data_manager = DataManager()
        from utils.config import DataConfig
        symbol = DataConfig.SYMBOLS["crypto"][0]
        df = data_manager.fetch_historical_data(symbol)
        if df.empty:
            logger.error(f"No data available for {symbol}")
        else:
            logger.info(f"Data available for {symbol}: {len(df)} records")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        return

    if args.train:
        logger.info("Starting model training...")
        train_models()
        if not args.pocket_ws and not args.combined and not args.ai:
            return

    if args.pocket_ws:
        logger.info("Starting Pocket Option WebSocket client...")
        asyncio.run(run_pocket_ws())
        return

    if args.combined:
        logger.info("Starting combined services...")
        asyncio.run(run_combined())
        return

    if args.ai:
        logger.info("Starting bot with AI Hook integration...")
        asyncio.run(run_ai_bot())
        return

    logger.info("Starting the Telegram bot...")
    run_bot()

if __name__ == "__main__":
    main()
