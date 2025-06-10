import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

class DataProcessor:
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.selected_features = None
        
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Data must contain OHLCV columns")
        
        # Add RSI
        data['rsi'] = ta.rsi(data['close'])
        
        # Add MACD
        macd = ta.macd(data['close'])
        data = pd.concat([data, macd], axis=1)
        
        # Add Bollinger Bands
        bbands = ta.bbands(data['close'])
        data = pd.concat([data, bbands], axis=1)
        
        # Add Stochastic Oscillator
        stoch = ta.stoch(data['high'], data['low'], data['close'])
        data = pd.concat([data, stoch], axis=1)
        
        # Add ATR
        data['atr'] = ta.atr(data['high'], data['low'], data['close'])
        
        # Add price momentum
        data['momentum'] = data['close'].pct_change(periods=5)
        
        # Add volume indicators
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_std'] = data['volume'].rolling(window=20).std()
        
        # Add price volatility
        data['volatility'] = data['close'].rolling(window=20).std()
        
        return data
    
    def create_target(self, data: pd.DataFrame, 
                     forward_period: int = 1) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            data: DataFrame with price data
            forward_period: Number of periods to look ahead
            
        Returns:
            DataFrame with added target column
        """
        # Calculate future returns
        future_returns = data['close'].shift(-forward_period) / data['close'] - 1
        
        # Create binary target (1 for price increase, 0 for decrease)
        data['target'] = (future_returns > 0).astype(int)
        
        return data
    
    def select_features(self, data: pd.DataFrame, 
                       target_col: str = 'target',
                       max_iter: int = 100) -> List[str]:
        """
        Select important features using Boruta algorithm.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            max_iter: Maximum number of iterations for Boruta
            
        Returns:
            List of selected feature names
        """
        # Prepare data
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Initialize Boruta
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
        
        # Fit Boruta
        boruta.fit(X.values, y.values)
        
        # Get selected features
        selected_features = X.columns[boruta.support_].tolist()
        self.selected_features = selected_features
        
        return selected_features
    
    def prepare_data(self, data: pd.DataFrame, 
                    sequence_length: int = 15) -> tuple:
        """
        Prepare data for LSTM model.
        
        Args:
            data: DataFrame with features and target
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if self.selected_features is None:
            raise ValueError("Features must be selected before preparing data")
        
        # Select features and scale
        features = data[self.selected_features]
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(scaled_features[i:(i + sequence_length)])
            y.append(data['target'].iloc[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def process_live_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process live data for prediction.
        
        Args:
            data: DataFrame with raw price data
            
        Returns:
            Processed DataFrame ready for prediction
        """
        # Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Select features
        if self.selected_features is None:
            raise ValueError("Features must be selected before processing live data")
        
        # Scale features
        features = data[self.selected_features]
        scaled_features = self.scaler.transform(features)
        
        return pd.DataFrame(scaled_features, columns=self.selected_features) 