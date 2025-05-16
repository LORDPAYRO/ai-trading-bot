"""
AI Hook for Telegram Trading Assistant Bot

This module provides the AI-powered analysis engine for the trading bot.
It continuously monitors market data, identifies strong trading opportunities,
and sends high-confidence signals to the Telegram bot.

Key Features:
- Advanced technical analysis using multiple indicators
- Pattern detection combining rule-based and ML approaches
- Pre-signal alerts sent before final trading signals
- Configurable parameters including confidence thresholds and alert delays

Components:
- analyzer.py: Market data analysis and pattern detection
- controller.py: Integration with WebSocket data and Telegram
- commands.py: Telegram command handlers for AI Hook
"""

__version__ = "1.1.0" 
