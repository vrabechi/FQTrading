import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from ..models.lstm_model import LSTMModel
from ..data.processor import DataProcessor

class FQTradingStrategy:
    def __init__(self, config_path: str = "../config/strategy_config.yaml",
                 model_path: Optional[str] = None):
        """
        Initialize the trading strategy with configuration.
        
        Args:
            config_path: Path to the strategy configuration file
            model_path: Path to saved LSTM model
        """
        self.config = self._load_config(config_path)
        self.positions = {}
        self.trades = []
        
        # Initialize LSTM model and data processor
        self.model = LSTMModel(sequence_length=15, model_path=model_path)
        self.data_processor = DataProcessor()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load strategy configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def train(self, data: pd.DataFrame, model_save_path: Optional[str] = None) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            data: DataFrame with historical price data
            model_save_path: Path to save the trained model
            
        Returns:
            Dictionary with training history
        """
        # Process data
        data = self.data_processor.add_technical_indicators(data)
        data = self.data_processor.create_target(data)
        
        # Select features
        self.data_processor.select_features(data)
        
        # Prepare data for training
        X, y = self.data_processor.prepare_data(data)
        
        # Train model
        history = self.model.train(
            X, y,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            model_save_path=model_save_path
        )
        
        return history
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the LSTM model predictions.
        
        Args:
            data: DataFrame with price data and indicators
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        
        # Process data
        processed_data = self.data_processor.process_live_data(data)
        
        # Create sequences for prediction
        sequence_length = self.model.sequence_length
        if len(processed_data) < sequence_length:
            return signals
        
        # Prepare sequences
        sequences = []
        for i in range(len(processed_data) - sequence_length + 1):
            sequences.append(processed_data.iloc[i:i + sequence_length].values)
        sequences = np.array(sequences)
        
        # Generate predictions
        predictions = self.model.predict(sequences)
        
        # Convert predictions to signals
        # Use a threshold of 0.5 for binary classification
        signal_indices = data.index[sequence_length - 1:]
        signals.loc[signal_indices, 'signal'] = np.where(predictions > 0.5, 1, -1)
        
        return signals
    
    def calculate_position_size(self, signal: int, price: float, 
                              portfolio_value: float) -> float:
        """
        Calculate the position size based on the signal and risk parameters.
        
        Args:
            signal: Trading signal (1: buy, -1: sell, 0: no action)
            price: Current price of the security
            portfolio_value: Current portfolio value
            
        Returns:
            Position size in units of the security
        """
        if signal == 0:
            return 0
            
        position_size = portfolio_value * self.config['trading']['position_size']
        return position_size / price
    
    def update_positions(self, signals: pd.DataFrame, 
                        prices: pd.DataFrame) -> None:
        """
        Update positions based on signals and current prices.
        
        Args:
            signals: DataFrame with trading signals
            prices: DataFrame with current prices
        """
        for timestamp, row in signals.iterrows():
            signal = row['signal']
            price = prices.loc[timestamp, 'close']
            
            if signal != 0:
                position_size = self.calculate_position_size(
                    signal, price, self._calculate_portfolio_value(price)
                )
                
                if signal == 1:  # Buy
                    self.positions[price] = position_size
                    self.trades.append({
                        'type': 'buy',
                        'price': price,
                        'size': position_size,
                        'timestamp': timestamp
                    })
                elif signal == -1:  # Sell
                    if self.positions:
                        for pos_price, pos_size in self.positions.items():
                            self.trades.append({
                                'type': 'sell',
                                'price': price,
                                'size': pos_size,
                                'timestamp': timestamp
                            })
                        self.positions.clear()
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        position_value = sum(size * current_price for size in self.positions.values())
        return position_value
    
    def generate_report(self) -> Dict:
        """
        Generate performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {}
            
        # Calculate metrics
        total_trades = len(self.trades)
        profitable_trades = sum(1 for trade in self.trades 
                              if trade['type'] == 'sell' and 
                              trade['price'] > trade['price'])
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'current_positions': len(self.positions)
        } 