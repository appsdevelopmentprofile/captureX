import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from streamlit_option_menu import option_menu  # Make sure to install streamlit-option-menu

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Multiple AI Improvements - RFO Central Application',
        [
            'Operations',
            "Workforce",
            "Compliance"
        ],
        menu_icon='hospital-fill',
        icons=['activity', 'people', 'check2-square'],
        default_index=0
    )

# Operations Page
if selected == 'Operations':
    st.title('Operations Module')

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ("Upstream", "Midstream", "Downstream"))

    # Function for the Upstream Section
    def upstream_section():
        st.header("Upstream Operations")
        st.write("This section focuses on exploration, extraction, and production activities.")
    
        # Streamlit app for Random Forest Regression
        st.title("Random Forest Regression with Feature Selection and Hyperparameter Tuning")
    
        # Step 1: Load the dataset
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
    
            # Step 2: Data Preprocessing
            data = data.dropna()
            target_column = st.text_input("Enter the target column name:", "MMP (mPa)")
    
            if target_column in data.columns:
                y = pd.to_numeric(data[target_column], errors='coerce')
                X = pd.get_dummies(data.drop(target_column, axis=1), drop_first=True)
    
                # Step 3: Feature scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
    
                # Feature selection using RFE with Cross-validation
                model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rfecv = RFECV(estimator=model_rf, step=1, cv=5, scoring='neg_mean_squared_error')
                rfecv.fit(X_scaled, y)
    
                selected_features = X.columns[rfecv.support_].tolist()
                st.write("Selected Features:")
                st.write(selected_features)
    
                # Step 4: Hyperparameter tuning with RandomizedSearchCV
                param_dist = {
                    'n_est
