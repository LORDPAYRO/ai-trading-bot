import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import ccxt
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

from utils.config import DataConfig, BINANCE_API_KEY, BINANCE_API_SECRET, ALPHA_VANTAGE_API_KEY
from utils.logger import logger

class DataFetcher(ABC):
    """Abstract base class for data fetchers"""
    
    @abstractmethod
    def fetch_historical_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Fetch historical price data for a symbol"""
        pass
    
    @abstractmethod
    def fetch_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch the latest price data for a symbol"""
        pass

class BinanceDataFetcher(DataFetcher):
    """Data fetcher for Binance cryptocurrency exchange"""
    
    def __init__(self):
        """Initialize the Binance data fetcher"""
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
        })
        logger.info("Initialized Binance data fetcher")
    
    def fetch_historical_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """
        Fetch historical price data from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Time interval (e.g., '1h', '1d')
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched historical data for {symbol} ({timeframe}): {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch the latest price data from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Time interval (e.g., '1h', '1d')
            
        Returns:
            DataFrame with the latest OHLCV data
        """
        try:
            # Fetch recent candles
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=3)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched latest data for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {str(e)}")
            return pd.DataFrame()

class AlphaVantageDataFetcher(DataFetcher):
    """Data fetcher for Alpha Vantage API (forex)"""
    
    def __init__(self):
        """Initialize the Alpha Vantage data fetcher"""
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        logger.info("Initialized Alpha Vantage data fetcher")
    
    def _map_timeframe(self, timeframe: str) -> str:
        """Map standard timeframe to Alpha Vantage interval"""
        mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '1d': 'daily',
        }
        return mapping.get(timeframe, '60min')
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Alpha Vantage API"""
        return symbol.replace('/', '')
    
    def fetch_historical_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """
        Fetch historical forex data from Alpha Vantage
        
        Args:
            symbol: Forex pair symbol (e.g., 'EUR/USD')
            timeframe: Time interval (e.g., '1h', '1d')
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            formatted_symbol = self._format_symbol(symbol)
            av_interval = self._map_timeframe(timeframe)
            
            # For daily data
            if av_interval == 'daily':
                params = {
                    'function': 'FX_DAILY',
                    'from_symbol': formatted_symbol[:3],
                    'to_symbol': formatted_symbol[3:],
                    'outputsize': 'full' if days > 100 else 'compact',
                    'apikey': self.api_key
                }
            # For intraday data
            else:
                params = {
                    'function': 'FX_INTRADAY',
                    'from_symbol': formatted_symbol[:3],
                    'to_symbol': formatted_symbol[3:],
                    'interval': av_interval,
                    'outputsize': 'full' if days > 100 else 'compact',
                    'apikey': self.api_key
                }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Get the time series data
            if 'Time Series FX (Daily)' in data:
                time_series = data['Time Series FX (Daily)']
            else:
                time_series_key = f"Time Series FX ({av_interval})"
                time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df.columns = [col.split('. ')[1] for col in df.columns]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Convert types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            # Set index as datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df = df.sort_index()
            
            # Filter for requested period
            start_date = datetime.now() - timedelta(days=days)
            df = df[df.index >= start_date]
            
            logger.info(f"Fetched historical data for {symbol} ({timeframe}): {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch the latest forex data from Alpha Vantage
        
        Args:
            symbol: Forex pair symbol (e.g., 'EUR/USD')
            timeframe: Time interval (e.g., '1h', '1d')
            
        Returns:
            DataFrame with the latest OHLCV data
        """
        try:
            # For forex, we can just get the last few records from historical data
            df = self.fetch_historical_data(symbol, timeframe, 1)
            
            logger.info(f"Fetched latest data for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {str(e)}")
            return pd.DataFrame()

class DataManager:
    """Manages data fetching operations for all symbols"""
    
    def __init__(self):
        """Initialize data fetchers"""
        self.binance_fetcher = BinanceDataFetcher()
        self.cached_data = {}  # Cache for historical data
        logger.info("Initialized Data Manager")
    
    def fetch_historical_data(self, symbol: str, timeframe: str = None, days: int = None) -> pd.DataFrame:
        """
        Fetch historical data for a symbol
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval (default from config)
            days: Number of days of historical data (default from config)
            
        Returns:
            DataFrame with OHLCV data
        """
        timeframe = timeframe or DataConfig.DEFAULT_TIMEFRAME
        days = days or DataConfig.LOOKBACK_PERIOD_DAYS
        
        # Check if data is in cache
        cache_key = f"{symbol}_{timeframe}_{days}"
        if cache_key in self.cached_data:
            data, timestamp = self.cached_data[cache_key]
            cache_age = datetime.now() - timestamp
            # If cache is less than 1 hour old, use it
            if cache_age.total_seconds() < 3600:
                logger.info(f"Using cached data for {symbol} ({timeframe})")
                return data
        
        # For now, assuming all symbols are crypto and use Binance
        data = self.binance_fetcher.fetch_historical_data(symbol, timeframe, days)
        
        # Update cache
        self.cached_data[cache_key] = (data, datetime.now())
        
        return data
    
    def fetch_latest_data(self, symbol: str, timeframe: str = None) -> pd.DataFrame:
        """
        Fetch the latest data for a symbol
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval (default from config)
            
        Returns:
            DataFrame with the latest OHLCV data
        """
        timeframe = timeframe or DataConfig.DEFAULT_TIMEFRAME
        
        # For now, assuming all symbols are crypto and use Binance
        return self.binance_fetcher.fetch_latest_data(symbol, timeframe)
    
    def fetch_all_historical_data(self, timeframe: str = None, days: int = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all configured symbols
        
        Args:
            timeframe: Time interval (default from config)
            days: Number of days of historical data (default from config)
            
        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        all_data = {}
        
        # For now, only fetch crypto symbols
        for symbol in DataConfig.SYMBOLS['crypto']:
            all_data[symbol] = self.fetch_historical_data(symbol, timeframe, days)
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        return all_data
    
    def fetch_all_latest_data(self, timeframe: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch the latest data for all configured symbols
        
        Args:
            timeframe: Time interval (default from config)
            
        Returns:
            Dictionary mapping symbols to DataFrames with the latest OHLCV data
        """
        all_data = {}
        
        # For now, only fetch crypto symbols
        for symbol in DataConfig.SYMBOLS['crypto']:
            all_data[symbol] = self.fetch_latest_data(symbol, timeframe)
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        return all_data

