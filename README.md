# FQ Trading Strategy

A flexible trading strategy framework that can be applied to different securities while maintaining the same core logic. The framework separates core trading functionality from reporting and visualization.

## Project Structure

```
FQTrading/
├── src/                      # Source code
│   ├── core/                # Core trading logic
│   │   ├── strategy.py      # Main strategy implementation
│   │   ├── position.py      # Position management
│   │   └── signals.py       # Signal generation
│   ├── data/                # Data management
│   │   ├── fetcher.py       # Data fetching utilities
│   │   └── processor.py     # Data processing and cleaning
│   ├── backtest/            # Backtesting framework
│   │   ├── engine.py        # Backtesting engine
│   │   └── metrics.py       # Performance metrics calculation
│   └── utils/               # Utility functions
├── config/                  # Configuration files
│   ├── strategy_config.yaml # Strategy parameters
│   └── securities.yaml      # Security definitions
├── data/                    # Data storage
│   ├── raw/                # Raw market data
│   └── processed/          # Processed data
├── reports/                # Reporting and visualization
│   ├── templates/          # Report templates
│   ├── generators/         # Report generation scripts
│   └── dashboards/         # Interactive dashboards
├── notebooks/              # Analysis notebooks
│   ├── strategy_dev/       # Strategy development
│   └── analysis/          # Performance analysis
└── tests/                 # Unit tests
```

## Core Features

- **Flexible Security Support**: Trade any security by configuring it in `config/securities.yaml`
- **Parameter Management**: Easily modify strategy parameters through configuration files
- **Backtesting**: Comprehensive backtesting framework with performance metrics
- **Position Management**: Robust position sizing and risk management
- **Signal Generation**: Clear separation of signal generation logic

## Reporting Features

- **Performance Reports**: Generate detailed performance reports
- **Position Reports**: Track and analyze position changes
- **Interactive Dashboards**: Visualize strategy performance
- **Custom Reports**: Create custom report templates

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure your securities and strategy parameters in the `config` directory
2. Run backtests using the backtesting framework
3. Generate reports using the reporting module
4. Deploy the strategy for live trading

## Development

- Core trading logic is isolated in the `src/core` directory
- Reporting functionality is separated in the `reports` directory
- Use notebooks in `notebooks/strategy_dev` for strategy development
- Use notebooks in `notebooks/analysis` for performance analysis 