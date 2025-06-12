import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import yaml

class DataProcessor:
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.selected_features = None
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from yaml file."""
        with open('config/strategy_config.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def _remove_correlated_features(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            data: DataFrame with features
            threshold: Correlation threshold above which features will be removed
            
        Returns:
            DataFrame with correlated features removed
        """
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation above threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Remove correlated features
        return data.drop(columns=to_drop)
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> pd.DataFrame:
        """
        Select features based on mutual information with the target.
        
        Args:
            X: DataFrame with features
            y: Target variable
            threshold: Minimum mutual information score to keep a feature
            
        Returns:
            DataFrame with only the selected features
        """
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y)
        
        # Create a Series with feature names and their scores
        mi_scores = pd.Series(mi_scores, index=X.columns)
        
        # Select features above threshold
        selected_features = mi_scores[mi_scores >= threshold].index
        
        return X[selected_features]
    
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
        
        df = data.copy()

        # Add all factors
        df.ta.strategy('All')
        df = df.bfill(axis=1)
        
        return df
    
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
                       max_iter: int = None) -> List[str]:
        """
        Select important features using correlation analysis, mutual information, and Boruta algorithm.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            max_iter: Maximum number of iterations for Boruta (overrides config if provided)
            
        Returns:
            List of selected feature names
        """
        # Prepare data
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Step 1: Remove correlated features
        correlation_threshold = self.config['feature_selection']['correlation_threshold']
        X = self._remove_correlated_features(X, correlation_threshold)
        
        # Step 2: Select features based on mutual information
        mutual_info_threshold = self.config['feature_selection']['mutual_info_threshold']
        X = self._select_by_mutual_info(X, y, mutual_info_threshold)
        
        # Step 3: Apply Boruta algorithm
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta = BorutaPy(rf, 
                         n_estimators='auto', 
                         verbose=2, 
                         random_state=42,
                         max_iter=max_iter or self.config['feature_selection']['boruta_max_iter'])
        
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