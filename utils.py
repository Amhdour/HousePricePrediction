import pandas as pd
import numpy as np

def load_sample_data(n_samples=1000):
    """Generate sample housing data"""
    np.random.seed(42)
    
    data = {
        'square_feet': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'lot_size': np.random.normal(8000, 2000, n_samples),
        'garage_spaces': np.random.randint(0, 4, n_samples),
        'price': np.zeros(n_samples)
    }
    
    # Generate prices based on features
    for i in range(n_samples):
        base_price = 200000
        price = (
            base_price +
            data['square_feet'][i] * 100 +
            data['bedrooms'][i] * 15000 +
            data['bathrooms'][i] * 20000 -
            data['age'][i] * 1000 +
            data['lot_size'][i] * 0.5 +
            data['garage_spaces'][i] * 10000
        )
        # Add some random noise
        data['price'][i] = price * np.random.normal(1, 0.1)
    
    df = pd.DataFrame(data)
    return df

def calculate_feature_importance(model, feature_columns):
    """Calculate and format feature importance"""
    importance = model.coef_
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': np.abs(importance)
    })
    importance_df = importance_df.sort_values('importance', ascending=True)
    return importance_df
