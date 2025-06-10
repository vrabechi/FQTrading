from datetime import datetime, timedelta
from src.run_strategy import main as run_strategy_main
import argparse

def create_args(mode: str, symbol: str, start_date: str = None, end_date: str = None, model_path: str = None):
    """Create argparse.Namespace object with the given parameters."""
    args = argparse.Namespace()
    args.mode = mode
    args.symbol = symbol
    args.start_date = start_date
    args.end_date = end_date
    args.model_path = model_path
    args.config_path = "../config/strategy_config.yaml"
    args.debug = True  # Always enable debug mode for debugging
    return args

def debug_train():
    # Set your parameters here
    symbol = "BTC-USD"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    model_path = f"../models/{symbol}_model.h5"
    
    # Create args object
    args = create_args(
        mode='train',
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        model_path=model_path
    )
    
    # Run strategy with debug parameters
    return run_strategy_main(args)

def debug_backtest():
    # Set your parameters here
    symbol = "BTC-USD"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    model_path = f"../models/{symbol}_model.h5"
    
    # Create args object
    args = create_args(
        mode='backtest',
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        model_path=model_path
    )
    
    # Run strategy with debug parameters
    return run_strategy_main(args)

def debug_live():
    # Set your parameters here
    symbol = "BTC-USD"
    model_path = f"../models/{symbol}_model.h5"
    
    # Create args object
    args = create_args(
        mode='live',
        symbol=symbol,
        model_path=model_path
    )
    
    # Run strategy with debug parameters
    return run_strategy_main(args)

if __name__ == "__main__":
    # Set breakpoints in your IDE and run one of these functions
    history = debug_train()
    # results = debug_backtest()
    # report = debug_live() 