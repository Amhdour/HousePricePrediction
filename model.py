import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

def perform_cross_validation(model, X, y, cv=5, scoring='r2'):
    """Perform cross-validation and return detailed metrics"""
    # Initialize KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Initialize arrays to store metrics
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    # Perform k-fold cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Train model on fold
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)

        # Calculate metrics for fold
        r2_scores.append(r2_score(y_val_fold, y_pred))
        mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))

    # Calculate mean and std for each metric
    cv_metrics = {
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores)
    }

    return cv_metrics

def train_model(df, feature_columns, target_column, model_type='linear', cv_folds=5):
    """Train the selected model and return model and metrics"""
    X = df[feature_columns]
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Select model based on type
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    else:  # default to linear regression
        model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate metrics on test set
    test_metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # Perform cross-validation
    cv_metrics = perform_cross_validation(model, X, y, cv=cv_folds)

    # Combine all metrics
    metrics = {
        'test': test_metrics,
        'cv': cv_metrics
    }

    return model, X_test, y_test, metrics

def predict_price(model, input_values, feature_columns):
    """Make a single prediction using the trained model"""
    input_data = np.array([[input_values[feature] for feature in feature_columns]])
    prediction = model.predict(input_data)[0]
    return prediction

def get_feature_importance(model, feature_columns, model_type='linear'):
    """Calculate and format feature importance based on model type"""
    if model_type == 'linear':
        importance = np.abs(model.coef_)
    elif model_type == 'random_forest':
        importance = model.feature_importances_
    elif model_type == 'xgboost':
        importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance
    })
    importance_df = importance_df.sort_values('importance', ascending=True)
    return importance_df

def compare_models(df, feature_columns, target_column, cv_folds=5):
    """Train and compare multiple models"""
    model_types = ['linear', 'random_forest', 'xgboost']
    comparison_results = {}

    for model_type in model_types:
        model, _, _, metrics = train_model(
            df, feature_columns, target_column, 
            model_type=model_type, cv_folds=cv_folds
        )
        comparison_results[model_type] = metrics

    return comparison_results