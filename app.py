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
from collections import defaultdict

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
    
                # Scatter Plot of Predicted vs. Actual Values
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
            else:
                st.error("Target column not found in the dataset. Please check the column name.")
        else:
            st.info("Please upload a CSV file to proceed.")

    # Function for the Midstream Section
    def midstream_section():
        st.header("Midstream Operations")
        st.write("This section includes transportation, storage, and handling.")

        # Simulated dataset including process, mechanical, and performance data
        @st.cache
        def load_data():
            data = pd.DataFrame({
                'flow': np.random.normal(100, 10, 1000),
                'level': np.random.normal(80, 8, 1000),
                'pressure': np.random.normal(150, 15, 1000),
                'temperature': np.random.normal(200, 20, 1000),
                'vibration': np.random.normal(0.5, 0.1, 1000),
                'bearing_temp': np.random.normal(50, 5, 1000),
                'lube_oil_temp': np.random.normal(60, 5, 1000),
                'surge_limit': np.random.normal(0.8, 0.05, 1000),
                'pump_performance': np.random.normal(75, 7, 1000),
                'compressor_curve': np.random.normal(0.9, 0.05, 1000),
                'failure_risk': np.random.normal(0.5, 0.2, 1000)
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
        def predictive_analytics(data):
            st.subheader("Predictive Analytics: Predict Asset Failures using Digital Twin")

            # Separate features (X) and target (y)
            X = data.drop('failure_risk', axis=1)
            y = data['failure_risk']

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fit model (Random Forest Regressor for demonstration)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)

            # Evaluate model
            st.write("Model Performance Metrics:")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

            # Feature importance
            importances = model.feature_importances_
            feature_importances = pd.DataFrame(importances, index=X.columns, columns=["Importance"]).sort_values("Importance", ascending=False)
            st.write("Feature Importance:")
            st.bar_chart(feature_importances)

        # Call the functions for analytics
        descriptive_analytics(data)
        diagnostic_analytics(data)
        predictive_analytics(data)

    # Function for the Downstream Section
    def downstream_section():
        st.header("Downstream Operations")
        st.write("This section covers refining, distribution, and retail.")  
        
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
    else:
        downstream_section()

# Operations Page
if selected == 'Workforce':
    st.title('Workforce Model')

    # Sidebar for OpenAI API credentials input
    with st.sidebar:
        st.header("🔑 OpenAI API Credentials")
        OPENAI_KEY = st.text_input("OpenAI API Key", type="password")
        ENDPOINT = st.text_input("API Endpoint", value="https://api.openai.com")
        OPENAI_VERSION = st.text_input("API Version", value="2023-03-15-preview")
    
    # Check if API key is provided
    if not OPENAI_KEY:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        # Sidebar for navigation
        selected = st.selectbox('OptiLabor Tool', 
                                 ['Virtual Assistant',
                                  'LaborSync: Automation (Augmented Reality + Digital Twins + On-Site Robots)',
                                  'Upskilling Module: Training & Competency'],
                                 index=0)

    # Front-End Layout
    st.title("💸 OptiLabor AI")
    st.markdown("Your AI assistant for industrial projects, providing real-time information, collaboration, and resource predictions.")
    
    # Display features
    st.markdown("### Features")
    features = [
        "OL Takeoff - Submit your blueprint for an accurate Bill of Quantities.",
        "OL Doc Wizard - Quickly find invoices, contracts, and create reports.",
        "OL Virtual Design - Identify design flaws early with 3D models.",
        "OL Project Management - Gain insights from real-time data using AI and IoT.",
        "OL Decision Making - Visualize KPIs and automate compliance reporting."
    ]
    st.markdown("\n".join(f"- {feature}" for feature in features))
    st.markdown("-------")
    
    # Display video and image
    st.video(open('Untitled video.mp4', 'rb').read())
    st.image('OptiLabor.png', caption='OptiLabor Features')
    
    # Virtual Assistant Module
    if selected == 'Virtual Assistant':
        st.title("Module 1: Virtual Assistant")
        st.markdown("An NLP assistant providing real-time information, reducing RFIs.")
        st.header("Document Management Challenges")
        st.image('docs construction.jpeg', caption='Source: InEight Blog')
    
        def init_page():
            st.header('OL DocWizard')
            st.write("I'm here to help you get information from your database documents.")
            st.sidebar.title('Options')
    
        def select_llm():
            return AzureOpenAI(model='gpt-35-turbo-16k', deployment_name='GPT35-optilabor', 
                               api_key=OPENAI_KEY, azure_endpoint=ENDPOINT, api_version=OPENAI_VERSION)
    
        def select_embedding():
            return AzureOpenAIEmbedding(model='text-embedding-ada-002', deployment_name='text-embedding-ada-002', 
                                         api_key=OPENAI_KEY, azure_endpoint=ENDPOINT, api_version=OPENAI_VERSION)
    
        def init_messages():
            if 'messages' not in st.session_state or st.sidebar.button('Clear Conversation'):
                st.session_state.messages = [SystemMessage(content='You are a helpful AI assistant. Reply in markdown format.')]
    
        def get_answer(query_engine, messages):
            response = query_engine.query(messages)
            return response.response, response.metadata
    
        def main():
            init_page()
            file = st.file_uploader('Upload file:', type=['pdf', 'txt', 'docx'])
            
            if file:
                with open(os.path.join('data', file.name), 'wb') as f: 
                    f.write(file.getbuffer())        
                st.success('Saved file!')
    
                documents = SimpleDirectoryReader('./data').load_data()
                file_names = [doc.metadata['file_name'] for doc in documents]
                st.write('Current documents in folder:', ', '.join(file_names))
    
                llm = select_llm()
                embed = select_embedding()
                service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed)
                query_engine = VectorStoreIndex.from_documents(documents, service_context=service_context).as_query_engine()
    
                init_messages()
    
                if user_input := st.chat_input('Input your question!'):
                    st.session_state.messages.append(HumanMessage(content=user_input))
                    with st.spinner('Bot is typing ...'):
                        answer, meta_data = get_answer(query_engine, user_input)
    
                    greetings = [...]  # Define greeting phrases
                    compliments = [...]  # Define compliment phrases
                    if user_input.lower() in greetings:
                        answer = 'Hi, how can I assist you?'
                    elif user_input.lower() in compliments:
                        answer = 'My pleasure! Feel free to ask more questions.'
                    elif any(keyword in answer for keyword in keywords):  # Define keywords
                        st.session_state.messages.append(AIMessage(content=f"**Source**: {list(meta_data.values())[0]['file_name']}  \n**Answer**: {answer}"))
                    else:
                        answer = 'This is outside the scope of the provided knowledge base.'
    
                    st.session_state.messages.append(AIMessage(content=answer))
    
                for message in st.session_state.get('messages', []):
                    with st.chat_message('assistant' if isinstance(message, AIMessage) else 'user'):
                        st.markdown(message.content)
            else:
                if not os.listdir('./data'):
                    st.write('No file is saved yet.')
    
        if __name__ == '__main__':
            main()
    
        # Reporting Visualization
        st.title("A lot of data is wasted")
        st.header("Construction projects generate massive amounts of data, yet 80% of it remains unstructured.")
        st.image('reporting.png', caption='Source: StructionSite Blog')
    
        # Show Power BI report
        path_to_html = "/Users/juanrivera/Desktop/chatbot1/Streamlit/pages/11_power_BI.html"
        with open(path_to_html, 'r') as f: 
            components.html(f.read(), height=2000)
    
    # Chatbot Report Generation
    st.markdown("<h1 style='text-align:justified;font-family:Georgia'>Construction Chatbot - Doc Generator</h1>", unsafe_allow_html=True)
    with st.sidebar:
        openai_api_key = st.secrets["auth_token"]
        st.markdown("-------")
        company_name = st.text_input("What is the name of the company?")
        start_up_description = st.text_input("Please describe your statement and objectives of your report")
        sector = st.multiselect('In which sector of construction is the project?', ["Non-Residential", "Residential"])
        st.markdown("-------")
        generate_button = st.button("Generate my Report")
    
    def generate_report(company_name, report_date):
        doc = docx.Document()
        doc.add_heading("Report", 0)
        doc.add_paragraph(f'Created On: {report_date}')
        doc.add_paragraph(f'Created For: {company_name}')
        doc.add_heading(f'Balance of {company_name} for {", ".join(sector)} sector')
        doc.save('Construction Report.docx')
        
        with open('Construction Report.docx', "rb") as file:
            data = file.read()
            encoded = base64.b64encode(data).decode('utf-8')
        st.download_button('Download Here', encoded, "Construction Report.docx")
    
    def generate_response(input_text):
        llm = OpenAI(temperature=0.3, openai_api_key=openai_api_key)
        return llm(input_text)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "", "content": "Hey there, I'm OptiLabor Bot, here to help you create your report. Please input your report details on the left."}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if generate_button:
        if openai_api_key.startswith('sk-'):
            date_today = datetime.date.today()
            funding_summary = generate_response(f"I'm exploring funding options for my startup '{company_name}'. The description is: {start_up_description}. Please provide an overview of different funding sources available to early-stage startups in the {', '.join(sector)} sector.")
            legal_summary = generate_response(f"I need legal guidance for launching '{company_name}' in the {', '.join(sector)} sector. Please provide relevant legal requirements and regulations.")
            
            generate_report(company_name, date_today)
        else:
            st.warning('Please enter your OpenAI API key!', icon='⚠')
    
    if (prompt := st.chat_input("What is up?")): 
        if openai_api_key.startswith('sk-'):
            st.session_state.messages.append({"role": "", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
    
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner('Wait for it...'):
                    assistant_response = generate_response(prompt)
                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "", "content": full_response})
        else:
            st.warning('Please enter your OpenAI API key!', icon='⚠')



# Compliance
if selected == 'Compliance':
    st.title("Compliance") 
