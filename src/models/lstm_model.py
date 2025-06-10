import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import os

class LSTMModel:
    def __init__(self, sequence_length: int = 15, model_path: Optional[str] = None):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            model_path: Path to saved model (if loading existing model)
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(units=50, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(units=50),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(units=1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        # Scale features
        features = self.scaler.fit_transform(data.drop('target', axis=1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(data['target'].iloc[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              model_save_path: Optional[str] = None) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X: Input features
            y: Target values
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_save_path: Path to save the trained model
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        if model_save_path:
            callbacks.append(
                ModelCheckpoint(model_save_path, save_best_only=True)
            )
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        return self.model.predict(X)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(path)
        
    def load_model(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        self.model = load_model(path) 