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

        import streamlit as st

    # Title for the Operations Module
    st.title("Operations Module")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ("Upstream", "Midstream", "Downstream"))

    # Function for the Upstream Section
    def upstream_section():
        st.header("Upstream Operations")
        st.write("This section focuses on exploration, extraction, and production activities.")
            import streamlit as st
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.feature_selection import RFECV
            from sklearn.model_selection import RandomizedSearchCV

            # Function to plot Taylor Diagram
            def plot_taylor_diagram(y_true, y_pred, title):
                corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
                std_obs = np.std(y_true)
                std_model = np.std(y_pred)

                plt.figure(figsize=(8, 8))
                plt.scatter(std_obs, corr_coef, color="b", marker="o", label="Model")
                plt.plot([0, std_obs], [corr_coef, corr_coef], "b--", label="Correlation")
                plt.plot([std_obs, std_obs], [0, corr_coef], "b--")
                plt.plot([0, std_obs], [0, corr_coef], "b--")

                plt.scatter(std_model, corr_coef, color="r", marker="o", label="Observations")
                plt.plot([0, std_model], [corr_coef, corr_coef], "r--", label="Correlation")
                plt.plot([std_model, std_model], [0, corr_coef], "r--")
                plt.plot([0, std_model], [0, corr_coef], "r--")

                plt.xlabel("Standard Deviation of Observations")
                plt.ylabel("Correlation Coefficient")
                plt.title(title)
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

            # Streamlit app
            st.title("Random Forest Regression with Feature Selection and Hyperparameter Tuning")

            # Step 1: Load the dataset
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)

                # Step 2: Data Preprocessing
                # Handle missing values
                data = data.dropna()

                # Ask user for target column name
                target_column = st.text_input("Enter the target column name:", "MMP (mPa)")

                if target_column in data.columns:
                    # Ensure target variable is numeric
                    y = pd.to_numeric(data[target_column], errors='coerce')

                    # Handle any categorical variables in X
                    X = pd.get_dummies(data.drop(target_column, axis=1), drop_first=True)

                    # Step 3: Feature scaling
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Feature selection using RFE with Cross-validation
                    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rfecv = RFECV(estimator=model_rf, step=1, cv=5, scoring='neg_mean_squared_error')
                    rfecv.fit(X_scaled, y)

                    # Get the selected features and their ranking
                    selected_features = X.columns[rfecv.support_].tolist()
                    st.write("Selected Features:")
                    st.write(selected_features)

                    # Step 4: Hyperparameter tuning with RandomizedSearchCV
                    param_dist = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 5, 10, 15],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                    }

                    model_rf_tuned = RandomForestRegressor(random_state=42)
                    rf_random = RandomizedSearchCV(estimator=model_rf_tuned,
                                                param_distributions=param_dist,
                                                n_iter=50,
                                                cv=5,
                                                scoring="neg_mean_squared_error",
                                                verbose=1,
                                                random_state=42)
                    rf_random.fit(X_scaled, y)

                    st.write("Best Hyperparameters:")
                    st.write(rf_random.best_params_)

                    # Step 5: Model Evaluation on Testing Set
                    X_selected = rfecv.transform(X_scaled)
                    X_train, X_test, y_train, y_test = train_test_split(X_selected,
                                                                        y,
                                                                        test_size=0.2,
                                                                        random_state=42)

                    model_rf_tuned_best = RandomForestRegressor(**rf_random.best_params_,
                                                                random_state=42)
                    model_rf_tuned_best.fit(X_train, y_train)
                    y_test_pred = model_rf_tuned_best.predict(X_test)

                    mse_test = mean_squared_error(y_test, y_test_pred)
                    mae_test = mean_absolute_error(y_test, y_test_pred)
                    r2_test = r2_score(y_test, y_test_pred)

                    st.write(f"Mean Squared Error (MSE) on Test Set: {mse_test}")
                    st.write(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")
                    st.write(f"R-squared (R2) Score on Test Set: {r2_test}")

                    # Step 6: Visualization

                    # Feature Importances Plot
                    feature_importances = model_rf_tuned_best.feature_importances_
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=selected_features,
                                y=feature_importances,
                                palette='viridis')
                    plt.xlabel("Selected Features")
                    plt.ylabel("Feature Importance")
                    plt.title("Feature Importance")
                    plt.tight_layout()
                    st.pyplot(plt)

                    # Residual Plot
                    residuals = y_test - y_test_pred
                    plt.scatter(y_test,
                                residuals,
                                alpha=0.7,
                                color='b')
                    plt.axhline(y=0,
                                color='k',
                                linestyle='--')
                    plt.xlabel("Actual MPG (mpg)")
                    plt.ylabel("Residuals (mpg)")
                    plt.title("Residual Plot")
                    plt.tight_layout()
                    st.pyplot(plt)

                    # Scatter Plot of Predicted vs. Actual Values with Confidence Interval
                    plt.scatter(y_test,
                                y_test_pred,
                                alpha=0.7,
                                color='b',
                                edgecolors='k')
                    plt.plot([min(y_test), max(y_test)],
                            [min(y_test), max(y_test)],
                            linestyle="--",
                            linewidth=2)
                    plt.xlabel("Actual MPG (mpg)")
                    plt.ylabel("Predicted MPG (mpg)")
                    plt.title("Scatter Plot of Predicted vs. Actual Values")
                    plt.tight_layout()
                    st.pyplot(plt)

                    # Step 7: Taylor Diagram
                    plot_taylor_diagram(y_test,
                                        y_test_pred,
                                        title="Taylor Diagram")
                else:
                    st.error("Target column not found in the dataset. Please check the column name.")
            else:
                st.info("Please upload a CSV file to proceed.")

    # Function for the Midstream Section
    def midstream_section():
        st.header("Midstream Operations")
        st.write("This section includes transportation, storage, and handling.")
            import streamlit as st
            import pandas as pd
            import numpy as np
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            from sklearn.decomposition import PCA

            # Set up the Streamlit app title and sidebar
            st.title("Southern Area Oil Operations - Data Analytics Platform")
            st.sidebar.header("Select Analytics Type")

            # Simulated dataset including process, mechanical, and performance data
            @st.cache
            def load_data():
                data = pd.DataFrame({
                    'flow': np.random.normal(100, 10, 1000),  # Process Model
                    'level': np.random.normal(80, 8, 1000),   # Process Model
                    'pressure': np.random.normal(150, 15, 1000),  # Process Model
                    'temperature': np.random.normal(200, 20, 1000),  # Process Model
                    'vibration': np.random.normal(0.5, 0.1, 1000),  # Mechanical Model
                    'bearing_temp': np.random.normal(50, 5, 1000),  # Mechanical Model
                    'lube_oil_temp': np.random.normal(60, 5, 1000),  # Mechanical Model
                    'surge_limit': np.random.normal(0.8, 0.05, 1000),  # Mechanical Model
                    'pump_performance': np.random.normal(75, 7, 1000),  # Performance Model
                    'compressor_curve': np.random.normal(0.9, 0.05, 1000),  # Performance Model
                    'failure_risk': np.random.normal(0.5, 0.2, 1000)  # Target for Predictive Analytics
                })
                return data

            data = load_data()

            # 1. Descriptive Analytics: Equipment Historical Data Overview
            def descriptive_analytics(data):
                st.subheader("Descriptive Analytics: Equipment Historical Data")
                st.write(data.describe())  # Display statistical summary

                # Visualize key features in dataset
                st.write("Data Visualization")
                sns.pairplot(data)
                st.pyplot(plt)

            # 2. Diagnostic Analytics: Root Cause Analysis using PCA and Drill Down
            def diagnostic_analytics(data):
                st.subheader("Diagnostic Analytics: Root Cause Analysis")
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data.drop('failure_risk', axis=1))  # Exclude target column

                st.write("Explained Variance by PCA Components:", pca.explained_variance_ratio_)

                # Scatter plot for PCA analysis
                fig, ax = plt.subplots()
                ax.scatter(pca_result[:, 0], pca_result[:, 1], c=data['failure_risk'], cmap='viridis')
                ax.set_title("PCA - Diagnostic Analysis")
                ax.set_xlabel("PCA1")
                ax.set_ylabel("PCA2")
                st.pyplot(fig)

            # 3. Predictive Analytics: Digital Twin and Predict Asset Failures
            def predictive_analytics(data, target_column):
                st.subheader("Predictive Analytics: Predict Asset Failures using Digital Twin")

                # Separate features (X) and target (y)
                X = data.drop(target_column, axis=1)
                y = data[target_column]

                # Split data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train Random Forest model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Predict on test set and calculate error
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                st.write(f"Mean Squared Error (MSE): {mse}")

                # Visualize prediction vs actual
                fig, ax = plt.subplots()
                ax.scatter(y_test, predictions, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel("Actual Failure Risk")
                ax.set_ylabel("Predicted Failure Risk")
                ax.set_title("Prediction vs Actual")
                st.pyplot(fig)

                return model

            # 4. Prescriptive Analytics: Action Recommendations based on Predictive Analytics
            def prescriptive_analytics(predicted_value, threshold=0.7):
                st.subheader("Prescriptive Analytics: Recommended Actions")
                if predicted_value > threshold:
                    st.write("High risk of failure detected. Recommended Action: Schedule Maintenance.")
                else:
                    st.write("Low risk of failure detected. Recommended Action: Continue Monitoring.")

            # Sidebar to select analytics type
            analytics_type = st.sidebar.radio(
                "Select the type of analytics to perform:",
                ('Descriptive', 'Diagnostic', 'Predictive', 'Prescriptive')
            )

            # Execute the selected analytics type
            if analytics_type == 'Descriptive':
                descriptive_analytics(data)

            elif analytics_type == 'Diagnostic':
                diagnostic_analytics(data)

            elif analytics_type == 'Predictive':
                model = predictive_analytics(data, 'failure_risk')
                simulated_predicted_value = model.predict([data.drop('failure_risk', axis=1).iloc[0]])
                st.write(f"Simulated Predicted Failure Risk: {simulated_predicted_value[0]}")

            elif analytics_type == 'Prescriptive':
                model = predictive_analytics(data, 'failure_risk')
                simulated_predicted_value = model.predict([data.drop('failure_risk', axis=1).iloc[0]])
                prescriptive_analytics(simulated_predicted_value[0])
            # Example: st.write("Midstream code output or data visualization")

    # Function for the Downstream Section
    def downstream_section():
        st.header("Downstream Operations")
        st.write("This section is for refining, processing, and distributing products.")
            import streamlit as st
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from collections import defaultdict

            # Placeholder function to simulate environment
            def simulate_investment(oil_gas_investment, renewables_investment, ccs_investment):
                # Simplified random simulation results
                production = oil_gas_investment * 0.5 + renewables_investment * 0.4
                co2_reduction = ccs_investment * 0.6
                return production, co2_reduction

            # Q-learning variables
            Q_table = defaultdict(lambda: np.zeros(3))  # Example state-action space
            alpha = 0.1  # Learning rate
            gamma = 0.95  # Discount factor
            epsilon = 0.1  # Exploration rate

            # Simulate a single step of Q-learning (simplified)
            def q_learning_step(state, action, reward, next_state):
                current_q_value = Q_table[state][action]
                next_max = np.max(Q_table[next_state])
                Q_table[state][action] = current_q_value + alpha * (reward + gamma * next_max - current_q_value)

            # Streamlit App
            st.title("Investment Decision Simulation")

            # File uploader for input
            uploaded_file = st.file_uploader("Upload your initial investment strategy (CSV)", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Investment Strategy", df)

            # Inputs for investments
            st.sidebar.title("Adjust Investment Levels")
            oil_gas_investment = st.sidebar.slider("Oil & Gas Investment", 0, 100, 50)
            renewables_investment = st.sidebar.slider("Renewables Investment", 0, 100, 50)
            ccs_investment = st.sidebar.slider("CCS Investment", 0, 100, 50)

            # Simulate and display results
            if st.sidebar.button("Simulate"):
                production, co2_reduction = simulate_investment(oil_gas_investment, renewables_investment, ccs_investment)
                st.write(f"Simulated Production: {production}")
                st.write(f"CO2 Reduction: {co2_reduction}")

                # Update Q-learning (simplified)
                state = (oil_gas_investment, renewables_investment, ccs_investment)
                action = np.random.choice([0, 1, 2])  # Random action for now
                reward = production - co2_reduction  # Simple reward metric
                next_state = state  # No transition in this step
                q_learning_step(state, action, reward, next_state)

                # Display updated Q-table
                st.write("Updated Q-table:", Q_table)

            # Placeholder for deep Q-learning section
            st.write("Deep Q-Network (DQN) implementation coming soon!")

            # Visualization section
            fig, ax = plt.subplots()
            ax.bar(["Oil & Gas", "Renewables", "CCS"], [oil_gas_investment, renewables_investment, ccs_investment])
            ax.set_ylabel("Investment Amount")
            st.pyplot(fig)


    # Display the selected section
    if section == "Upstream":
        upstream_section()
    elif section == "Midstream":
        midstream_section()
    elif section == "Downstream":
        downstream_section()


# Workforce Page
elif selected == 'Workforce':
    st.title('Workforce Module')

    # Add relevant content and functionalities for the Workforce module here

# Compliance Page
elif selected == 'Compliance':
    st.title('Compliance Module')

    # Add relevant content and functionalities for the Compliance module here
