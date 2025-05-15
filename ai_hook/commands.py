from telegram import Update
from telegram.ext import ContextTypes
import asyncio
from typing import Dict, Any, List, Optional

from utils.config import TelegramConfig
from utils.logger import logger
from ai_hook.controller import AIHookController

# Global controller instance
ai_controller = None

async def init_ai_hook(telegram_bot, symbols=None):
    """
    Initialize the AI Hook with the provided Telegram bot
    
    Args:
        telegram_bot: Telegram bot instance
        symbols: List of symbols to monitor
    
    Returns:
        AIHookController instance
    """
    global ai_controller
    
    if ai_controller is None:
        ai_controller = AIHookController(telegram_bot)
    
    if not ai_controller.is_running:
        success = await ai_controller.start(symbols)
        if not success:
            logger.error("Failed to initialize AI Hook")
            return None
    
    return ai_controller

async def ai_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /ai_start command to start the AI Hook"""
    user_id = update.effective_user.id
    if user_id not in TelegramConfig.ADMIN_USER_IDS:
        await update.message.reply_text("You are not authorized to use this command.")
        return
    
    await update.message.reply_text("Starting AI Hook... Please wait.")
    
    try:
        # Get symbols from command arguments if provided
        symbols = None
        if context.args:
            symbols = context.args
        
        # Initialize the AI Hook
        controller = await init_ai_hook(context.bot, symbols)
        
        if controller:
            symbols_text = ", ".join(controller.active_symbols) if controller.active_symbols else "none"
            await update.message.reply_text(
                f"AI Hook started successfully!\n\n"
                f"Monitoring symbols: {symbols_text}\n\n"
                f"Use /ai_status to check the AI status."
            )
        else:
            await update.message.reply_text("Failed to start AI Hook. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Error in ai_start_command: {str(e)}")
        await update.message.reply_text(f"Error starting AI Hook: {str(e)}")

async def ai_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /ai_stop command to stop the AI Hook"""
    user_id = update.effective_user.id
    if user_id not in TelegramConfig.ADMIN_USER_IDS:
        await update.message.reply_text("You are not authorized to use this command.")
        return
    
    global ai_controller
    if ai_controller and ai_controller.is_running:
        await update.message.reply_text("Stopping AI Hook... Please wait.")
        
        try:
            await ai_controller.stop()
            await update.message.reply_text("AI Hook stopped successfully.")
            
        except Exception as e:
            logger.error(f"Error stopping AI Hook: {str(e)}")
            await update.message.reply_text(f"Error stopping AI Hook: {str(e)}")
    else:
        await update.message.reply_text("AI Hook is not running.")

async def ai_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /ai_status command to check AI Hook status"""
    user_id = update.effective_user.id
    if user_id not in TelegramConfig.ADMIN_USER_IDS:
        await update.message.reply_text("You are not authorized to use this command.")
        return
    
    global ai_controller
    if ai_controller:
        try:
            # Get AI Hook stats
            stats = await ai_controller.get_stats()
            
            # Format accuracy
            accuracy = stats["accuracy"]
            accuracy_text = f"Overall accuracy: {accuracy['overall']*100:.2f}%\n"
            accuracy_text += f"Total signals: {accuracy['total_signals']}\n"
            accuracy_text += f"Correct signals: {accuracy['correct_signals']}\n\n"
            
            if accuracy['by_symbol']:
                accuracy_text += "*Accuracy by symbol:*\n"
                for symbol, acc in accuracy['by_symbol'].items():
                    accuracy_text += f"{symbol}: {acc*100:.2f}%\n"
            
            # Format active symbols
            symbols_text = ", ".join(stats["active_symbols"]) if stats["active_symbols"] else "none"
            
            # Pre-alert information
            pre_alert_text = f"Delay: {stats['pre_alert_delay']} seconds\n"
            pre_alert_text += f"Pending alerts: {stats['pending_pre_alerts']}\n"
            
            # Format message
            message = f"""
*AI Hook Status*

Running: {'Yes' if ai_controller.is_running else 'No'}
Active symbols: {symbols_text}
Signals in queue: {stats['signals_in_queue']}
Confidence threshold: {ai_controller.analyzer.min_confidence*100:.0f}%
Scan interval: {ai_controller.scan_interval} seconds
Cooldown period: {ai_controller.cooldown_period/3600:.1f} hours

*Pre-Alert Settings:*
{pre_alert_text}

*Performance:*
{accuracy_text}

Use /ai_symbols to manage monitored symbols.
Use /ai_config to adjust settings.
"""
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error getting AI Hook status: {str(e)}")
            await update.message.reply_text(f"Error retrieving AI Hook status: {str(e)}")
    else:
        await update.message.reply_text(
            "AI Hook is not initialized. Use /ai_start to start monitoring."
        )

async def ai_symbols_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /ai_symbols command to manage monitored symbols"""
    user_id = update.effective_user.id
    if user_id not in TelegramConfig.ADMIN_USER_IDS:
        await update.message.reply_text("You are not authorized to use this command.")
        return
    
    global ai_controller
    if not ai_controller:
        await update.message.reply_text("AI Hook is not initialized. Use /ai_start first.")
        return
    
    # Check if we have arguments (add/remove)
    if context.args and len(context.args) >= 2:
        action = context.args[0].lower()
        symbol = context.args[1].upper()
        
        if action == "add":
            success = await ai_controller.add_symbol(symbol)
            if success:
                await update.message.reply_text(f"Added {symbol} to monitored symbols.")
            else:
                await update.message.reply_text(f"Failed to add {symbol}. Check logs for details.")
                
        elif action == "remove":
            success = await ai_controller.remove_symbol(symbol)
            if success:
                await update.message.reply_text(f"Removed {symbol} from monitored symbols.")
            else:
                await update.message.reply_text(f"Failed to remove {symbol}. Check logs for details.")
                
        else:
            await update.message.reply_text(
                "Invalid action. Use `/ai_symbols add SYMBOL` or `/ai_symbols remove SYMBOL`."
            )
            
    else:
        # Just show the current symbols
        symbols = list(ai_controller.active_symbols)
        symbols_text = "\n".join(symbols) if symbols else "No symbols being monitored."
        
        message = f"""
*Currently Monitored Symbols:*

{symbols_text}

*Commands:*
- Add symbol: `/ai_symbols add SYMBOL`
- Remove symbol: `/ai_symbols remove SYMBOL`

Example: `/ai_symbols add BTC/USD`
"""
        await update.message.reply_text(message, parse_mode="Markdown")

async def ai_config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /ai_config command to adjust AI Hook settings"""
    user_id = update.effective_user.id
    if user_id not in TelegramConfig.ADMIN_USER_IDS:
        await update.message.reply_text("You are not authorized to use this command.")
        return
    
    global ai_controller
    if not ai_controller:
        await update.message.reply_text("AI Hook is not initialized. Use /ai_start first.")
        return
    
    # Check if we have arguments
    if context.args and len(context.args) >= 2:
        setting = context.args[0].lower()
        value = context.args[1]
        
        if setting == "confidence":
            try:
                confidence = float(value)
                if 0 <= confidence <= 1:
                    ai_controller.set_signal_confidence_threshold(confidence)
                    await update.message.reply_text(f"Signal confidence threshold set to {confidence*100:.0f}%")
                else:
                    await update.message.reply_text("Confidence must be between 0 and 1.")
            except ValueError:
                await update.message.reply_text("Invalid value. Confidence must be a number between 0 and 1.")
                
        elif setting == "interval":
            try:
                interval = int(value)
                if interval > 0:
                    ai_controller.set_scan_interval(interval)
                    await update.message.reply_text(f"Scan interval set to {interval} seconds")
                else:
                    await update.message.reply_text("Interval must be positive.")
            except ValueError:
                await update.message.reply_text("Invalid value. Interval must be a positive integer.")
                
        elif setting == "cooldown":
            try:
                hours = float(value)
                if hours >= 0:
                    ai_controller.set_cooldown_period(hours)
                    await update.message.reply_text(f"Signal cooldown period set to {hours} hours")
                else:
                    await update.message.reply_text("Cooldown period must be non-negative.")
            except ValueError:
                await update.message.reply_text("Invalid value. Cooldown must be a non-negative number.")
                
        elif setting == "alert_delay":
            try:
                seconds = int(value)
                if seconds >= 10:
                    ai_controller.set_pre_alert_delay(seconds)
                    await update.message.reply_text(f"Pre-alert delay set to {seconds} seconds")
                else:
                    await update.message.reply_text("Alert delay must be at least 10 seconds.")
            except ValueError:
                await update.message.reply_text("Invalid value. Alert delay must be a positive integer.")
                
        else:
            await update.message.reply_text(
                "Invalid setting. Available settings: confidence, interval, cooldown, alert_delay."
            )
            
    else:
        # Show current configuration
        message = f"""
*Current AI Hook Configuration:*

- Confidence threshold: {ai_controller.analyzer.min_confidence*100:.0f}%
- Scan interval: {ai_controller.scan_interval} seconds
- Cooldown period: {ai_controller.cooldown_period/3600:.1f} hours
- Pre-alert delay: {ai_controller.pre_alert_delay} seconds

*Commands:*
- Set confidence: `/ai_config confidence 0.9`
- Set scan interval: `/ai_config interval 30`
- Set cooldown period: `/ai_config cooldown 1.5`
- Set pre-alert delay: `/ai_config alert_delay 60`
"""
        await update.message.reply_text(message, parse_mode="Markdown")

async def ai_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /ai_help command to show AI Hook commands"""
    user_id = update.effective_user.id
    if user_id not in TelegramConfig.ADMIN_USER_IDS:
        await update.message.reply_text("You are not authorized to use this command.")
        return
    
    message = """
*AI Hook Commands:*

- `/ai_start [SYMBOL1 SYMBOL2 ...]` - Start the AI Hook (optional symbols)
- `/ai_stop` - Stop the AI Hook
- `/ai_status` - Check AI Hook status and performance
- `/ai_symbols` - Manage monitored symbols
- `/ai_config` - Adjust AI settings
- `/ai_help` - Show this help message

*Examples:*
- `/ai_start BTC/USD ETH/USD` - Start with specific symbols
- `/ai_symbols add XRP/USD` - Add a new symbol
- `/ai_config confidence 0.95` - Set confidence threshold to 95%
- `/ai_config alert_delay 30` - Set pre-alert delay to 30 seconds
"""
    
    await update.message.reply_text(message, parse_mode="Markdown")

def register_ai_commands(application):
    """
    Register AI Hook commands with the Telegram application
    
    Args:
        application: Telegram ApplicationBuilder instance
    """
    from telegram.ext import CommandHandler
    
    # Register command handlers
    application.add_handler(CommandHandler("ai_start", ai_start_command))
    application.add_handler(CommandHandler("ai_stop", ai_stop_command))
    application.add_handler(CommandHandler("ai_status", ai_status_command))
    application.add_handler(CommandHandler("ai_symbols", ai_symbols_command))
    application.add_handler(CommandHandler("ai_config", ai_config_command))
    application.add_handler(CommandHandler("ai_help", ai_help_command))
    
    logger.info("Registered AI Hook commands") 