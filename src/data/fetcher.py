import yaml
import pandas as pd
import yfinance as yf
import ccxt
from typing import Dict, Optional
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, securities_config_path: str = "config/securities.yaml"):
        """
        Initialize the data fetcher with securities configuration.
        
        Args:
            securities_config_path: Path to the securities configuration file
        """
        self.config = self._load_config(securities_config_path)
        self.exchanges = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load securities configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _get_exchange(self, exchange_name: str) -> ccxt.Exchange:
        """Get or create exchange instance."""
        if exchange_name not in self.exchanges:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchanges[exchange_name] = exchange_class()
        return self.exchanges[exchange_name]
    
    def fetch_data(self, symbol: str, start_date: Optional[str] = None,
                  end_date: Optional[str] = None, interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical data for a given symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with historical data
        """
        if symbol not in self.config['securities']:
            raise ValueError(f"Symbol {symbol} not found in configuration")
            
        security_config = self.config['securities'][symbol]
        data_source = security_config['data_source']
        
        if data_source == 'yfinance':
            df = self._fetch_yfinance_data(symbol, start_date, end_date, interval)
        elif data_source == 'ccxt':
            df = self._fetch_ccxt_data(symbol, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        # Standardize columns to ['open', 'high', 'low', 'close', 'volume']
        col_map = {c: c.lower() for c in df.columns}
        df.rename(columns=col_map, inplace=True)

        return df
    
    def _fetch_yfinance_data(self, symbol: str, start_date: Optional[str],
                            end_date: Optional[str], interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        return ticker.history(start=start_date, end=end_date, interval=interval)
    
    def _fetch_ccxt_data(self, symbol: str, start_date: Optional[str],
                         end_date: Optional[str], interval: str) -> pd.DataFrame:
        """Fetch data from CCXT exchange."""
        security_config = self.config['securities'][symbol]
        exchange = self._get_exchange(security_config['exchange'])
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, interval, start_timestamp, end_timestamp)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df 