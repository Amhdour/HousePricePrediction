import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import train_model, predict_price, get_feature_importance, compare_models
from utils import load_sample_data
from database import save_prediction, get_recent_predictions
import time
from feature_engineering import FeatureEngineer, get_available_transformations
from model_deployment import ModelManager
import json

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title and introduction
st.title("🏠 House Price Prediction App")
st.markdown("""
This app predicts house prices based on various features using machine learning.
Upload your own dataset or use our sample data to get started!
""")

# Sidebar
st.sidebar.header("Configuration")

# Data loading
data_option = st.sidebar.radio(
    "Choose data source",
    ["Use Sample Data", "Upload Your Own Data"]
)

if data_option == "Use Sample Data":
    df = load_sample_data()
    st.sidebar.success("Sample data loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error("Error: Please upload a valid CSV file")
            st.stop()
    else:
        st.info("Please upload a dataset or select 'Use Sample Data'")
        st.stop()

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Training",
    "Model Comparison",
    "Price Prediction",
    "Prediction History",
    "Model Deployment"
])

with tab1:
    st.header("📊 Data Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sample Data")
        st.dataframe(df.head())

    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())

    # Feature Selection
    st.header("🎯 Feature Selection")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    target_column = st.selectbox("Select Target Column (Price)", numeric_columns)
    feature_columns = st.multiselect(
        "Select Features for Prediction",
        [col for col in numeric_columns if col != target_column],
        default=[col for col in numeric_columns if col != target_column][:3]
    )

    if not feature_columns:
        st.warning("Please select at least one feature for prediction.")
        st.stop()

    # Feature Engineering
    st.header("🛠️ Feature Engineering")
    transformations = get_available_transformations()

    # Scaling options
    scaling_type = st.selectbox(
        "Select Scaling Method",
        transformations['scaling'],
        help="StandardScaler normalizes features to zero mean and unit variance. MinMaxScaler scales features to a fixed range."
    )

    # Polynomial features
    poly_degree = st.selectbox(
        "Polynomial Features Degree",
        transformations['polynomial'],
        help="Create polynomial features up to the specified degree. Higher degrees capture more complex relationships but may overfit."
    )

    # Log transformation
    if transformations['log_transform']:
        log_transform_cols = st.multiselect(
            "Select Columns for Log Transformation",
            feature_columns,
            help="Apply log transformation to handle skewed distributions"
        )

    # Interaction terms
    if transformations['interactions'] and len(feature_columns) > 1:
        st.subheader("Feature Interactions")
        interaction_cols = st.multiselect(
            "Select Features for Interaction Terms",
            feature_columns,
            max_selections=2,
            help="Create interaction terms between selected features"
        )

    # Model Selection
    st.header("🤖 Model Selection")
    model_type = st.selectbox(
        "Select Model Type",
        ['linear', 'random_forest', 'xgboost'],
        format_func=lambda x: {
            'linear': 'Linear Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost'
        }[x]
    )

    # Cross-validation configuration
    cv_folds = st.slider("Number of Cross-validation Folds", min_value=3, max_value=10, value=5)

    # Model Training
    if st.button("Train Model"):
        with st.spinner("Engineering features and training model..."):
            # Create a copy of the features
            X = df[feature_columns].copy()

            # Apply selected transformations
            if scaling_type != 'none':
                X = st.session_state.feature_engineer.apply_scaling(X, scaling_type)

            if poly_degree > 1:
                X = st.session_state.feature_engineer.create_polynomial_features(X, poly_degree)

            if log_transform_cols:
                X = st.session_state.feature_engineer.apply_log_transform(X, log_transform_cols)

            if len(interaction_cols) == 2:
                X = st.session_state.feature_engineer.create_interaction_terms(
                    X, [(interaction_cols[0], interaction_cols[1])]
                )

            # Train model with engineered features
            model, X_test, y_test, metrics = train_model(
                pd.concat([X, df[[target_column]]], axis=1),
                X.columns.tolist(),
                target_column,
                model_type=model_type,
                cv_folds=cv_folds
            )

            st.session_state.model = model
            st.session_state.feature_columns = X.columns.tolist()
            st.session_state.model_type = model_type

            # Display test metrics
            st.subheader("📊 Test Set Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{metrics['test']['r2']:.3f}")
            col2.metric("MAE", f"{metrics['test']['mae']:.2f}")
            col3.metric("RMSE", f"{metrics['test']['rmse']:.2f}")

            # Display cross-validation metrics
            st.subheader("🎯 Cross-validation Results")
            cv_metrics = metrics['cv']
            col1, col2, col3 = st.columns(3)
            col1.metric("CV R² Score", f"{cv_metrics['r2_mean']:.3f} (±{cv_metrics['r2_std']:.3f})")
            col2.metric("CV MAE", f"{cv_metrics['mae_mean']:.2f} (±{cv_metrics['mae_std']:.2f})")
            col3.metric("CV RMSE", f"{cv_metrics['rmse_mean']:.2f} (±{cv_metrics['rmse_std']:.2f})")

            # Feature importance
            st.subheader("📈 Feature Importance")
            importance_df = get_feature_importance(
                model, X.columns.tolist(), model_type=model_type
            )
            fig_importance = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance'
            )
            st.plotly_chart(fig_importance)

            # Save the trained model
            version = st.session_state.model_manager.save_model(
                model=model,
                model_type=model_type,
                feature_columns=X.columns.tolist(),
                metrics=metrics
            )
            st.success(f"Model trained and saved as version {version}")


with tab2:
    st.header("🔄 Model Comparison")

    if st.button("Compare All Models"):
        with st.spinner("Training and comparing models..."):
            comparison_results = compare_models(
                df, feature_columns, target_column, cv_folds=cv_folds
            )
            st.session_state.comparison_results = comparison_results

            # Create comparison visualizations
            st.subheader("📊 Model Performance Comparison")

            # Prepare data for visualization
            model_names = []
            r2_scores = []
            r2_stds = []
            mae_scores = []
            mae_stds = []
            rmse_scores = []
            rmse_stds = []

            for model_type, metrics in comparison_results.items():
                model_names.append(model_type.replace('_', ' ').title())
                r2_scores.append(metrics['cv']['r2_mean'])
                r2_stds.append(metrics['cv']['r2_std'])
                mae_scores.append(metrics['cv']['mae_mean'])
                mae_stds.append(metrics['cv']['mae_std'])
                rmse_scores.append(metrics['cv']['rmse_mean'])
                rmse_stds.append(metrics['cv']['rmse_std'])

            # Create tabs for different visualizations
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "Score Comparison",
                "Detailed Metrics",
                "Cross-validation Analysis",
                "Performance Summary"
            ])

            with viz_tab1:
                # R² Score comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='R² Score',
                    x=model_names,
                    y=r2_scores,
                    error_y=dict(type='data', array=r2_stds),
                    marker_color='rgb(55, 83, 109)'
                ))

                fig.update_layout(
                    title='Model Comparison - R² Score (higher is better)',
                    xaxis_title='Model',
                    yaxis_title='R² Score',
                    barmode='group'
                )
                st.plotly_chart(fig)

                # MAE comparison
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Bar(
                    name='MAE',
                    x=model_names,
                    y=mae_scores,
                    error_y=dict(type='data', array=mae_stds),
                    marker_color='rgb(219, 64, 82)'
                ))

                fig_mae.update_layout(
                    title='Model Comparison - Mean Absolute Error (lower is better)',
                    xaxis_title='Model',
                    yaxis_title='MAE',
                    barmode='group'
                )
                st.plotly_chart(fig_mae)

                # RMSE comparison
                fig_rmse = go.Figure()
                fig_rmse.add_trace(go.Bar(
                    name='RMSE',
                    x=model_names,
                    y=rmse_scores,
                    error_y=dict(type='data', array=rmse_stds),
                    marker_color='rgb(50, 171, 96)'
                ))

                fig_rmse.update_layout(
                    title='Model Comparison - Root Mean Square Error (lower is better)',
                    xaxis_title='Model',
                    yaxis_title='RMSE',
                    barmode='group'
                )
                st.plotly_chart(fig_rmse)

            with viz_tab2:
                # Create detailed metrics table
                st.subheader("📋 Detailed Metrics Table")
                metrics_data = []
                for model_type, metrics in comparison_results.items():
                    metrics_data.append({
                        'Model': model_type.replace('_', ' ').title(),
                        'R² Score (mean)': f"{metrics['cv']['r2_mean']:.3f}",
                        'R² Score (std)': f"±{metrics['cv']['r2_std']:.3f}",
                        'MAE (mean)': f"{metrics['cv']['mae_mean']:.2f}",
                        'MAE (std)': f"±{metrics['cv']['mae_std']:.2f}",
                        'RMSE (mean)': f"{metrics['cv']['rmse_mean']:.2f}",
                        'RMSE (std)': f"±{metrics['cv']['rmse_std']:.2f}"
                    })
                st.dataframe(pd.DataFrame(metrics_data))

                # Add metrics explanation
                st.markdown("""
                ### 📝 Metrics Explanation
                - **R² Score**: Coefficient of determination (higher is better)
                  - Ranges from 0 to 1, where 1 indicates perfect prediction
                - **MAE**: Mean Absolute Error (lower is better)
                  - Average absolute difference between predicted and actual values
                - **RMSE**: Root Mean Square Error (lower is better)
                  - Square root of the average squared differences
                  - Penalizes larger errors more heavily than MAE
                """)

            with viz_tab3:
                # Cross-validation analysis
                st.subheader("🎯 Cross-validation Performance Analysis")

                # Create cross-validation plots
                for metric in ['R² Score', 'MAE', 'RMSE']:
                    fig_cv = go.Figure()

                    for model_type, metrics in comparison_results.items():
                        model_name = model_type.replace('_', ' ').title()

                        # Fix metric key mapping
                        if metric == 'R² Score':
                            metric_key = 'r2'
                        else:
                            metric_key = metric.lower()

                        mean_val = metrics['cv'][f'{metric_key}_mean']
                        std_val = metrics['cv'][f'{metric_key}_std']

                        # Add range area
                        fig_cv.add_trace(go.Scatter(
                            name=f'{model_name} Range',
                            x=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
                            y=[mean_val] * 5,
                            error_y=dict(
                                type='data',
                                array=[std_val] * 5,
                                visible=True
                            ),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))

                        # Add mean line
                        fig_cv.add_trace(go.Scatter(
                            name=model_name,
                            x=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
                            y=[mean_val] * 5,
                            mode='lines+markers',
                        ))

                    fig_cv.update_layout(
                        title=f'Cross-validation {metric} Across Folds',
                        xaxis_title='Fold',
                        yaxis_title=metric,
                        showlegend=True
                    )
                    st.plotly_chart(fig_cv)

            with viz_tab4:
                # Performance summary
                st.subheader("📊 Performance Summary")

                # Find best model for each metric
                best_models = {
                    'R² Score': max(
                        model_names,
                        key=lambda x: r2_scores[model_names.index(x)]
                    ),
                    'MAE': min(
                        model_names,
                        key=lambda x: mae_scores[model_names.index(x)]
                    ),
                    'RMSE': min(
                        model_names,
                        key=lambda x: rmse_scores[model_names.index(x)]
                    )
                }

                # Display best models
                st.markdown("### 🏆 Best Performing Models")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Best R² Score",
                        best_models['R² Score'],
                        f"{max(r2_scores):.3f}"
                    )

                with col2:
                    st.metric(
                        "Best MAE",
                        best_models['MAE'],
                        f"{min(mae_scores):.2f}"
                    )

                with col3:
                    st.metric(
                        "Best RMSE",
                        best_models['RMSE'],
                        f"{min(rmse_scores):.2f}"
                    )

                # Add recommendation
                st.markdown("### 💡 Model Recommendation")
                # Count how many times each model is best
                best_model_counts = {}
                for model in best_models.values():
                    best_model_counts[model] = best_model_counts.get(model, 0) + 1

                recommended_model = max(best_model_counts.items(), key=lambda x: x[1])[0]
                st.success(f"""
                Based on the comprehensive analysis, the recommended model is: **{recommended_model}**

                This recommendation is based on:
                - Overall performance across all metrics
                - Consistency in cross-validation results
                - Balance between accuracy and reliability
                """)


with tab3:
    st.header("💰 Price Prediction")

    # Get currently deployed model
    current_model, current_features, model_version = st.session_state.model_manager.get_current_model()

    if current_model is None:
        st.warning("No model is currently deployed. Please deploy a model first.")
    else:
        st.success(f"Using deployed model version: {model_version}")
        st.markdown("Enter values for selected features to predict house price:")

        input_values = {}
        col1, col2 = st.columns(2)
        for i, feature in enumerate(current_features):
            with col1 if i % 2 == 0 else col2:
                input_values[feature] = st.number_input(
                    f"Enter {feature}",
                    value=float(df[feature].mean()) if feature in df.columns else 0.0,
                    format="%.2f"
                )

        if st.button("Predict Price"):
            with st.spinner("Calculating prediction..."):
                prediction = predict_price(current_model, input_values, current_features)
                st.success(f"Predicted House Price: ${prediction:,.2f}")

                # Save prediction to database
                prediction_data = {
                    **{k: float(v) for k, v in input_values.items()},
                    'predicted_price': float(prediction)
                }
                save_prediction(prediction_data)

with tab4:
    st.header("📜 Prediction History")
    recent_predictions = get_recent_predictions()

    if recent_predictions:
        predictions_df = pd.DataFrame([
            {
                'Date': pred.created_at,
                'Predicted Price': f"${pred.predicted_price:,.2f}",
                'Square Feet': pred.square_feet,
                'Bedrooms': pred.bedrooms,
                'Bathrooms': pred.bathrooms,
                'Age': pred.age,
                'Lot Size': pred.lot_size,
                'Garage Spaces': pred.garage_spaces
            }
            for pred in recent_predictions
        ])
        st.dataframe(predictions_df)
    else:
        st.info("No predictions have been made yet.")

with tab5:
    st.header("🚀 Model Deployment Management")

    # Model deployment history
    st.subheader("📜 Deployment History")
    deployments = st.session_state.model_manager.get_deployment_history()

    if deployments:
        deployment_data = []
        for dep in deployments:
            metrics = json.loads(dep.metrics)
            cv_metrics = metrics.get('cv', {})
            deployment_data.append({
                'Version': dep.model_version,
                'Model Type': dep.model_type,
                'R² Score': f"{cv_metrics.get('r2_mean', 0):.3f}",
                'Status': '🟢 Deployed' if dep.deployed else '⚪ Not Deployed',
                'Created At': dep.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'Deployed At': dep.deployed_at.strftime('%Y-%m-%d %H:%M:%S') if dep.deployed_at else '-'
            })

        df_deployments = pd.DataFrame(deployment_data)
        st.dataframe(df_deployments)

        # Model deployment section
        st.subheader("🔄 Deploy Model")
        versions = [dep.model_version for dep in deployments]
        selected_version = st.selectbox(
            "Select Model Version to Deploy",
            versions,
            format_func=lambda x: f"Version {x} ({next(d['Model Type'] for d in deployment_data if d['Version'] == x)})"
        )

        if st.button("Deploy Selected Model"):
            with st.spinner("Deploying model..."):
                if st.session_state.model_manager.deploy_model(selected_version):
                    st.success(f"Successfully deployed model version {selected_version}")
                else:
                    st.error("Failed to deploy model")
    else:
        st.info("No models have been trained and saved yet. Train a model to see deployment options.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)