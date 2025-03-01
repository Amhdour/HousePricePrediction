import pandas as pd
import numpy as np

def generate_sample_data(n_samples=1000):
    """Generate sample housing data for testing"""
    np.random.seed(42)
    
    data = {
        'square_feet': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'lot_size': np.random.normal(8000, 2000, n_samples),
        'garage_spaces': np.random.randint(0, 4, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df
