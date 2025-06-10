import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from ..core.strategy import FQTradingStrategy

class BacktestEngine:
    def __init__(self, strategy: FQTradingStrategy):
        """
        Initialize the backtesting engine.
        
        Args:
            strategy: Trading strategy instance
        """
        self.strategy = strategy
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        self.returns = []
        
    def run(self, data: pd.DataFrame, initial_capital: float) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with historical price data
            initial_capital: Initial portfolio value
            
        Returns:
            Dictionary with backtest results
        """
        self.portfolio_value = [initial_capital]
        self.returns = []
        current_capital = initial_capital
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        # Iterate through each time period
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Calculate position size
            position_size = self.strategy.calculate_position_size(
                signal, current_price, current_capital
            )
            
            # Update positions
            if signal != 0:
                self._update_position(signal, position_size, current_price)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.portfolio_value.append(portfolio_value)
            
            # Calculate returns
            returns = (portfolio_value - current_capital) / current_capital
            self.returns.append(returns)
            current_capital = portfolio_value
        
        return self._generate_results()
    
    def _update_position(self, signal: int, size: float, price: float) -> None:
        """Update position based on signal."""
        if signal == 1:  # Buy
            self.positions[price] = size
            self.trades.append({
                'type': 'buy',
                'price': price,
                'size': size,
                'timestamp': datetime.now()
            })
        elif signal == -1:  # Sell
            if self.positions:
                for pos_price, pos_size in self.positions.items():
                    self.trades.append({
                        'type': 'sell',
                        'price': price,
                        'size': pos_size,
                        'timestamp': datetime.now()
                    })
                self.positions.clear()
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        position_value = sum(size * current_price for size in self.positions.values())
        return position_value
    
    def _generate_results(self) -> Dict:
        """Generate backtest results."""
        returns_series = pd.Series(self.returns)
        portfolio_series = pd.Series(self.portfolio_value)
        
        return {
            'total_return': (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0],
            'sharpe_ratio': self._calculate_sharpe_ratio(returns_series),
            'max_drawdown': self._calculate_max_drawdown(portfolio_series),
            'num_trades': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'portfolio_value': portfolio_series.tolist(),
            'returns': returns_series.tolist()
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = portfolio_value.expanding().max()
        drawdowns = (portfolio_value - rolling_max) / rolling_max
        return abs(drawdowns.min())
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if not self.trades:
            return 0.0
            
        profitable_trades = sum(1 for trade in self.trades 
                              if trade['type'] == 'sell' and 
                              trade['price'] > trade['price'])
        return profitable_trades / len(self.trades) 