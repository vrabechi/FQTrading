import argparse
import yaml
import pandas as pd
from datetime import datetime, timedelta
from src.data.fetcher import DataFetcher
from src.core.strategy import FQTradingStrategy
from src.backtest.engine import BacktestEngine
from src.reports.generators.performance_report import PerformanceReport

def parse_args():
    parser = argparse.ArgumentParser(description='Run FQ Trading Strategy')
    parser.add_argument('--mode', choices=['train', 'backtest', 'live'],
                       required=True, help='Strategy mode')
    parser.add_argument('--symbol', required=True, help='Trading symbol')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--model-path', help='Path to saved model')
    parser.add_argument('--config-path', default='../config/strategy_config.yaml',
                       help='Path to strategy config')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed logging')
    return parser.parse_args()

def main(args=None):
    """
    Main function to run the trading strategy.
    
    Args:
        args: Optional argparse.Namespace object. If None, arguments will be parsed from command line.
    """
    if args is None:
        args = parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize components
    data_fetcher = DataFetcher()
    strategy = FQTradingStrategy(
        config_path=args.config_path,
        model_path=args.model_path
    )
    
    if args.mode == 'train':
        # Fetch training data
        data = data_fetcher.fetch_data(
            args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Train model
        history = strategy.train(
            data,
            model_save_path=f'../models/{args.symbol}_model.h5'
        )
        
        print("Training completed. Model saved.")
        return history
        
    elif args.mode == 'backtest':
        # Fetch backtest data
        data = data_fetcher.fetch_data(
            args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Initialize backtest engine
        backtest = BacktestEngine(strategy)
        
        # Run backtest
        results = backtest.run(
            data,
            initial_capital=config['backtest']['initial_capital']
        )
        
        # Generate report
        report = PerformanceReport(results)
        report.generate_report()
        
        print("Backtest completed. Report generated.")
        return results
        
    elif args.mode == 'live':
        # Fetch latest data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = data_fetcher.fetch_data(
            args.symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Update positions
        strategy.update_positions(signals, data)
        
        # Generate report
        report = strategy.generate_report()
        
        print("Live trading signals generated:")
        print(f"Current positions: {report['current_positions']}")
        print(f"Total trades: {report['total_trades']}")
        print(f"Win rate: {report['win_rate']:.2%}")
        return report

if __name__ == '__main__':
    main() 