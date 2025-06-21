import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import RobustScaler
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.cluster import KMeans
import yaml

class DataProcessor:
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = RobustScaler()
        self.selected_features = None
        self.config = self._load_config()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.kmeans = None
        self.cluster_centers = None
        
    def _load_config(self) -> dict:
        """Load configuration from yaml file."""
        with open('config/strategy_config.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def _split_data(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation and test sets.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Get split ratios from config
        train_ratio = self.config['feature_selection']['data_split']['train_ratio']
        val_ratio = self.config['feature_selection']['data_split']['val_ratio']
        test_ratio = self.config['feature_selection']['data_split']['test_ratio']
        random_state = self.config['feature_selection']['data_split']['random_state']
        
        # First split: separate out the test set
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_ratio,
            random_state=random_state,
            stratify=data[target_col],
            shuffle=False
        )
        
        # Second split: separate training and validation sets
        # Adjust val_ratio to account for the reduced dataset size
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=adjusted_val_ratio,
            random_state=random_state,
            stratify=train_val_data[target_col],
            shuffle=False
        )
        
        return train_data, val_data, test_data
    
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
    
    def _apply_kmeans(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply K-Means clustering to reduce dimensionality.
        
        Args:
            X: DataFrame with features
            
        Returns:
            DataFrame with cluster assignments
        """
        num_clusters = self.config['feature_selection']['clustering']['num_clusters']
        
        # If clustering is disabled, return original features
        if num_clusters < 1:
            return X
        
        # Initialize and fit K-Means
        self.kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=self.config['feature_selection']['clustering']['random_state'],
            max_iter=self.config['feature_selection']['clustering']['max_iter'],
            n_init=self.config['feature_selection']['clustering']['n_init']
        )
        
        # Fit K-Means and get cluster assignments
        cluster_assignments = self.kmeans.fit_predict(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Create new features based on distance to cluster centers
        cluster_features = pd.DataFrame(index=X.index)
        for i in range(num_clusters):
            # Calculate distance to each cluster center
            distances = np.linalg.norm(X - self.cluster_centers[i], axis=1)
            cluster_features[f'cluster_{i}_dist'] = distances
        
        return cluster_features
    
    def select_features(self, data: pd.DataFrame, 
                       target_col: str = 'target',
                       max_iter: int = None) -> List[str]:
        """
        Select important features using correlation analysis, mutual information, 
        Boruta algorithm, and optional K-Means clustering.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            max_iter: Maximum number of iterations for Boruta (overrides config if provided)
            
        Returns:
            List of selected feature names
        """
        # Step 1: Split data into train/val/test sets
        self.train_data, self.val_data, self.test_data = self._split_data(data, target_col)
        
        # Step 2: Prepare training data
        X_train = self.train_data.drop(target_col, axis=1)
        y_train = self.train_data[target_col]
        
        # Step 3: Remove correlated features from training data
        correlation_threshold = self.config['feature_selection']['correlation_threshold']
        X_train = self._remove_correlated_features(X_train, correlation_threshold)
        
        # Step 4: Select features based on mutual information
        mutual_info_threshold = self.config['feature_selection']['mutual_info_threshold']
        X_train = self._select_by_mutual_info(X_train, y_train, mutual_info_threshold)
        
        # Step 5: Apply Boruta algorithm
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        boruta = BorutaPy(rf, 
                         n_estimators='auto', 
                         verbose=2, 
                         random_state=42,
                         perc=10,
                         max_iter=max_iter or self.config['feature_selection']['boruta_max_iter'])
        
        # Fit Boruta on training data
        boruta.fit(X_train.values, y_train.values)
        
        # Get selected features
        selected_features = X_train.columns[boruta.support_].tolist()
        X_train_selected = X_train[selected_features]
        
        # Step 6: Apply K-Means clustering if enabled
        X_train_final = self._apply_kmeans(X_train_selected)
        self.selected_features = X_train_final.columns.tolist()
        
        # Apply the same transformations to validation and test sets
        X_val = self.val_data[selected_features]
        X_test = self.test_data[selected_features]
        
        if self.kmeans is not None:
            # For validation and test sets, use the same cluster centers
            X_val_final = pd.DataFrame(index=X_val.index)
            X_test_final = pd.DataFrame(index=X_test.index)
            
            for i in range(len(self.cluster_centers)):
                # Calculate distances using the same cluster centers
                val_distances = np.linalg.norm(X_val - self.cluster_centers[i], axis=1)
                test_distances = np.linalg.norm(X_test - self.cluster_centers[i], axis=1)
                
                X_val_final[f'cluster_{i}_dist'] = val_distances
                X_test_final[f'cluster_{i}_dist'] = test_distances
            
            self.val_data = pd.concat([X_val_final, self.val_data[[target_col]]], axis=1)
            self.test_data = pd.concat([X_test_final, self.test_data[[target_col]]], axis=1)
        else:
            self.val_data = pd.concat([X_val, self.val_data[[target_col]]], axis=1)
            self.test_data = pd.concat([X_test, self.test_data[[target_col]]], axis=1)
        
        return self.selected_features
    
    def get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the train, validation, and test data splits.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise ValueError("Data splits not available. Call select_features first.")
        return self.train_data, self.val_data, self.test_data
    
    def prepare_data(self, data: pd.DataFrame = None) -> Dict[str, TimeseriesGenerator]:
        """
        Prepare data for time series model using TimeseriesGenerator.
        If data is provided, it will be used instead of stored splits.
        
        Args:
            data: Optional DataFrame to use instead of stored splits
            
        Returns:
            Dictionary containing train, validation, and test generators
        """
        if data is None:
            if self.train_data is None or self.val_data is None or self.test_data is None:
                raise ValueError("Data splits not available. Call select_features first.")
            train_data = self.train_data
            val_data = self.val_data
            test_data = self.test_data
        else:
            # If data is provided, use it for all splits (useful for live data)
            train_data = val_data = test_data = data
        
        # Get sequence parameters from config
        sequence_length = self.config['feature_selection']['sequence']['length']
        stride = self.config['feature_selection']['sequence']['stride']
        sampling_rate = self.config['feature_selection']['sequence']['sampling_rate']
        
        # Prepare features and target
        def prepare_split(split_data):
            features = split_data[self.selected_features]
            target = split_data['target']
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features) if data is None else self.scaler.transform(features)
            
            return scaled_features, target.values
        
        # Prepare each split
        X_train, y_train = prepare_split(train_data)
        X_val, y_val = prepare_split(val_data)
        X_test, y_test = prepare_split(test_data)
        
        # Create generators
        self.train_generator = TimeseriesGenerator(
            X_train, y_train,
            length=sequence_length,
            stride=stride,
            sampling_rate=sampling_rate,
            batch_size=32
        )
        
        self.val_generator = TimeseriesGenerator(
            X_val, y_val,
            length=sequence_length,
            stride=stride,
            sampling_rate=sampling_rate,
            batch_size=32
        )
        
        self.test_generator = TimeseriesGenerator(
            X_test, y_test,
            length=sequence_length,
            stride=stride,
            sampling_rate=sampling_rate,
            batch_size=32
        )
        
        return {
            'train': self.train_generator,
            'val': self.val_generator,
            'test': self.test_generator
        }
    
    def process_live_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Process live data for prediction.
        
        Args:
            data: DataFrame with raw price data
            
        Returns:
            Processed data ready for prediction
        """
        # Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Select features
        if self.selected_features is None:
            raise ValueError("Features must be selected before processing live data")
        
        # Scale features
        features = data[self.selected_features]
        scaled_features = self.scaler.transform(features)
        
        # Create a single sequence for prediction
        sequence_length = self.config['feature_selection']['sequence']['length']
        if len(scaled_features) < sequence_length:
            raise ValueError(f"Live data must have at least {sequence_length} time steps")
            
        # Get the last sequence
        last_sequence = scaled_features[-sequence_length:]
        
        # Reshape for model input (add batch dimension)
        return last_sequence.reshape(1, sequence_length, -1) 