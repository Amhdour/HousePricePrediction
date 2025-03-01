import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureEngineer:
    def __init__(self):
        self.scaler = None
        self.poly_features = None
        self.feature_names = None
        
    def apply_scaling(self, X, scaler_type='standard'):
        """Apply scaling to features"""
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:  # minmax
            self.scaler = MinMaxScaler()
            
        return pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
    
    def create_polynomial_features(self, X, degree=2, include_bias=False):
        """Create polynomial features"""
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        feature_names = X.columns
        
        # Generate polynomial feature names
        poly_features = self.poly_features.fit_transform(X)
        self.feature_names = self.poly_features.get_feature_names_out(feature_names)
        
        return pd.DataFrame(
            poly_features,
            columns=self.feature_names
        )
    
    def apply_log_transform(self, X, columns):
        """Apply log transformation to specified columns"""
        X_transformed = X.copy()
        for col in columns:
            # Add small constant to handle zeros
            X_transformed[f"log_{col}"] = np.log1p(X_transformed[col])
        return X_transformed
    
    def create_interaction_terms(self, X, feature_pairs):
        """Create interaction terms between specified feature pairs"""
        X_transformed = X.copy()
        for feat1, feat2 in feature_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                X_transformed[f"{feat1}_{feat2}_interaction"] = X[feat1] * X[feat2]
        return X_transformed

def get_available_transformations():
    """Return list of available feature transformations"""
    return {
        'scaling': ['none', 'standard', 'minmax'],
        'polynomial': [1, 2, 3],
        'log_transform': True,
        'interactions': True
    }
