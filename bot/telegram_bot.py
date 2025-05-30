﻿from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram import Update
import asyncio
import datetime

from utils.config import TelegramConfig, TELEGRAM_API_TOKEN
from utils.logger import logger
from ai_hook.commands import register_ai_commands

class TelegramBot:
    """Telegram bot for sending trading signals"""
    
    def __init__(self, token=None):
        """Initialize the Telegram bot with the given token"""
        self.token = token or TELEGRAM_API_TOKEN
        self.application = ApplicationBuilder().token(self.token).build()
        self.setup_handlers()
        logger.info("Telegram bot initialized")
    
    def setup_handlers(self):
        """Set up command handlers for the bot"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("signal", self.signal_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("list", self.list_command))
        
        # Register AI Hook commands
        register_ai_commands(self.application)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        await update.message.reply_text(
            "👋 Welcome to the AI Trading Assistant Bot!\n\n"
            "I can help you find the best trading opportunities across multiple markets.\n\n"
            "Use /help to see available commands."
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command"""
        commands = "\n".join([f"/{cmd} - {desc}" for cmd, desc in TelegramConfig.COMMANDS.items()])
        
        # Add AI Hook commands to help
        ai_commands = """
        
*AI Hook Commands:*
/ai_help - Show AI Hook commands
/ai_start - Start AI market analysis
/ai_status - Check AI status and performance
/ai_symbols - Manage monitored symbols
/ai_config - Configure AI settings
/ai_stop - Stop AI market analysis
"""
        
        await update.message.reply_text(
            f"📚 Available commands:\n\n{commands}\n{ai_commands}",
            parse_mode="Markdown"
        )
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /analyze command"""
        await update.message.reply_text("🔍 Analyzing markets... This feature will be implemented soon.")
    
    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /signal command"""
        await update.message.reply_text("🔍 Finding the strongest trading signal... Feature coming soon.")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /status command"""
        await update.message.reply_text("Bot is currently in development mode.")
    
    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /list command"""
        from utils.config import DataConfig
        
        # Format the list of monitored symbols
        message = "📋 Monitored Symbols:\n\n"
        
        for category, symbols in DataConfig.SYMBOLS.items():
            message += f"{category.upper()}:\n"
            for symbol in symbols:
                message += f"- {symbol}\n"
            message += "\n"
        
        await update.message.reply_text(message)
    
    def run(self):
        """Run the bot"""
        logger.info("Starting the Telegram bot")
        self.application.run_polling()

if __name__ == "__main__":
    bot = TelegramBot()
    bot.run()
