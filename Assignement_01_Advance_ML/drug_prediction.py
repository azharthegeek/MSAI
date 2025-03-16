import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import base64
import io

# Set page configuration
st.set_page_config(
    page_title="Drug Classification App",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and introduction
st.title("AI5003 - Advanced Machine Learning - Drug Classification")
st.markdown("### By: Muhammad Azhar (24K-7606)")
st.markdown("### Submitted to: Professor Dr. Muhammad Rafi")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a page", 
    ["Dataset Overview", "Exploratory Data Analysis", "Model Training & Evaluation", "Make Predictions"]
)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset for A1P2 drug200.csv')
        # Convert object columns using the built-in str type for Arrow compatibility
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to preprocess data
@st.cache_data
def preprocess_data(df):
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)
    
    # Map drug categories
    drug_mapping = {
        'drugY': 0,
        'drugC': 1,
        'drugX': 2,
        'drugA': 3,
        'drugB': 4
    }
    df_encoded['Drug_encoded'] = df_encoded['Drug'].map(drug_mapping)
    
    # Prepare features and target
    X = df_encoded.drop(['Drug', 'Drug_encoded'], axis=1)
    y = df_encoded['Drug_encoded']
    
    # Normalize numerical features
    scaler = StandardScaler()
    X[['Age', 'Na_to_K']] = scaler.fit_transform(X[['Age', 'Na_to_K']])
    
    return X, y, drug_mapping

# Load data
df = load_data()

if df is not None:
    # Dataset Overview page
    if page == "Dataset Overview":
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Preview of the dataset")
            st.dataframe(df.head())
        
        with col2:
            st.subheader("Dataset Information")
            # Create a more Streamlit-friendly way to display DataFrame info
            buffer = io.StringIO()
            df.info(verbose=True, buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include='all'))
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        
        st.subheader("Categorical Column Values")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Sex categories:", df['Sex'].unique())
        with col2:
            st.write("BP categories:", df['BP'].unique())
        with col3:
            st.write("Cholesterol categories:", df['Cholesterol'].unique())
        st.write("Drug categories:", df['Drug'].unique())

    # Exploratory Data Analysis page
    elif page == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        st.subheader("Distribution of Drug Categories")
        fig, ax = plt.subplots(figsize=(10, 3))
        drug_counts = df['Drug'].value_counts()
        sns.barplot(x=drug_counts.index, y=drug_counts.values, color='green', ax=ax)
        plt.title("Distribution of Drug Categories")
        plt.xlabel("Drug Type")
        plt.ylabel("Count")
        st.pyplot(fig)
        
        st.subheader("Age Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df["Age"], bins=20, kde=True, color='green', ax=ax)
            plt.title("Age Distribution")
            plt.xlabel("Age")
            plt.ylabel("Frequency")
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Drug", y="Age", data=df, color='green', ax=ax)
            plt.title("Age by Drug Category")
            st.pyplot(fig)
            
        st.subheader("Na_to_K Ratio")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['Na_to_K'], bins=30, kde=True, color='green', ax=ax)
            plt.title("Distribution of Na_to_K Ratio")
            plt.xlabel("Na_to_K")
            plt.ylabel("Frequency")
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x="Drug", y="Na_to_K", data=df, color='green', ax=ax)
            plt.title("Na_to_K Ratio by Drug Category")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        st.subheader("Na_to_K Statistics")
        st.write(df['Na_to_K'].describe())
            
        st.subheader("Relationship between Categorical Variables and Drug")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.countplot(x='Drug', hue='Sex', data=df, palette='dark:green', ax=axes[0])
        axes[0].set_title("Drug Distribution by Sex")
        axes[0].set_xlabel("Drug")
        axes[0].set_ylabel("Count")
        
        custom_palette = ["red", "green", "black"]
        sns.countplot(x='Drug', hue='BP', data=df, palette=custom_palette, ax=axes[1])
        axes[1].set_title("Drug Distribution by Blood Pressure")
        axes[1].set_xlabel("Drug")
        axes[1].set_ylabel("Count")
        
        sns.countplot(x='Drug', hue='Cholesterol', data=df, palette='dark:green', ax=axes[2])
        axes[2].set_title("Drug Distribution by Cholesterol")
        axes[2].set_xlabel("Drug")
        axes[2].set_ylabel("Count")
        
        plt.tight_layout()
        st.pyplot(fig)

    # Model Training & Evaluation page
    elif page == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")
        
        # Preprocess data
        X, y, drug_mapping = preprocess_data(df)
        
        # Model selection and hyperparameters
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 1, 100, 42)
        with col2:
            dt_max_depth = st.slider("Decision Tree Max Depth", 1, 10, 3)
            rf_n_estimators = st.slider("Random Forest # of Trees", 10, 200, 100)
            rf_max_depth = st.slider("Random Forest Max Depth", 1, 10, 3)
        
        # Train models
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                st.success(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
                
                # Decision Tree
                st.subheader("Decision Tree Model")
                dt_classifier = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=random_state)
                dt_classifier.fit(X_train, y_train)
                dt_y_pred = dt_classifier.predict(X_test)
                
                dt_accuracy = metrics.accuracy_score(y_test, dt_y_pred)
                dt_cv_scores = cross_val_score(dt_classifier, X, y, cv=5)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Decision Tree Accuracy", f"{dt_accuracy:.4f}")
                    st.write(f"Cross-Validation Score: {dt_cv_scores.mean():.4f} ± {dt_cv_scores.std():.4f}")
                    
                # Reverse the drug mapping for display
                reverse_mapping = {v: k for k, v in drug_mapping.items()}
                label_names = [reverse_mapping[i] for i in range(len(drug_mapping))]
                    
                with col2:
                    st.write("Classification Report:")
                    report = classification_report(y_test, dt_y_pred, 
                                                target_names=label_names,
                                                output_dict=True, 
                                                zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format('{:.2f}'))
                
                # Decision Tree Confusion Matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, dt_y_pred)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=label_names,
                            yticklabels=label_names, ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix - Decision Tree')
                st.pyplot(fig)
                
                # Decision Tree visualization
                st.write("Decision Tree Visualization:")
                fig, ax = plt.subplots(figsize=(12, 8))
                tree.plot_tree(dt_classifier,
                            feature_names=X.columns,
                            class_names=label_names,
                            filled=True,
                            rounded=True, ax=ax)
                st.pyplot(fig)
                
                # Feature importance for Decision Tree
                dt_feature_imp = pd.Series(dt_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 5))
                dt_feature_imp.plot.bar(color='green', ax=ax)
                plt.title('Feature Importance in Decision Tree Model')
                plt.ylabel('Feature Importance Score')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Random Forest
                st.subheader("Random Forest Model")
                rf_classifier = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state)
                rf_classifier.fit(X_train, y_train)
                rf_y_pred = rf_classifier.predict(X_test)
                
                rf_accuracy = metrics.accuracy_score(y_test, rf_y_pred)
                rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Random Forest Accuracy", f"{rf_accuracy:.4f}")
                    st.write(f"Cross-Validation Score: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
                    
                with col2:
                    st.write("Classification Report:")
                    report = classification_report(y_test, rf_y_pred, 
                                                target_names=label_names,
                                                output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format('{:.2f}'))
                
                # Random Forest Confusion Matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, rf_y_pred)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=label_names,
                            yticklabels=label_names, ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix - Random Forest')
                st.pyplot(fig)
                
                # Sample tree from Random Forest
                st.write("Sample Tree from Random Forest:")
                fig, ax = plt.subplots(figsize=(12, 8))
                tree.plot_tree(rf_classifier.estimators_[0],
                            feature_names=X.columns,
                            class_names=label_names,
                            filled=True,
                            rounded=True, ax=ax)
                st.pyplot(fig)
                
                # Feature importance for Random Forest
                rf_feature_imp = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 5))
                rf_feature_imp.plot.bar(color='green', ax=ax)
                plt.title('Feature Importance in Random Forest Model')
                plt.ylabel('Feature Importance Score')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Model Comparison
                st.subheader("Model Comparison")
                models = ['Decision Tree', 'Random Forest']
                accuracies = [dt_accuracy, rf_accuracy]
                cv_scores = [dt_cv_scores.mean(), rf_cv_scores.mean()]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bar_width = 0.35
                x = np.arange(len(models))
                plt.bar(x - bar_width/2, accuracies, bar_width, label='Test Accuracy', color='green')
                plt.bar(x + bar_width/2, cv_scores, bar_width, label='CV Accuracy', color='black')
                plt.ylabel('Accuracy')
                plt.title('Model Performance Comparison')
                plt.xticks(x, models)
                plt.legend()
                plt.ylim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Compare feature importance
                st.write("Feature Importance Comparison:")
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    dt_feature_imp.plot.bar(color='green', ax=ax)
                    plt.title('Feature Importance - Decision Tree')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    rf_feature_imp.plot.bar(color='blue', ax=ax)
                    plt.title('Feature Importance - Random Forest')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Save models for prediction
                st.session_state['dt_model'] = dt_classifier
                st.session_state['rf_model'] = rf_classifier
                st.session_state['feature_columns'] = X.columns
                st.session_state['drug_mapping'] = drug_mapping

    # Make Predictions page
    elif page == "Make Predictions":
        st.header("Make Predictions for New Patients")
        
        # Check if models have been trained
        if 'dt_model' not in st.session_state or 'rf_model' not in st.session_state:
            st.warning("Please train the models first on the 'Model Training & Evaluation' page.")
        else:
            # Create input form for patient data
            st.subheader("Enter Patient Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=15, max_value=80, value=35)
                sex = st.selectbox("Sex", options=["F", "M"])
            
            with col2:
                bp = st.selectbox("Blood Pressure", options=["HIGH", "LOW", "NORMAL"])
                cholesterol = st.selectbox("Cholesterol", options=["HIGH", "NORMAL"])
            
            with col3:
                na_to_k = st.number_input("Na_to_K Ratio", min_value=6.0, max_value=40.0, value=15.0, step=0.1)
            
            # Model selection
            model_choice = st.radio("Choose a model for prediction", ["Decision Tree", "Random Forest"])
            
            if st.button("Predict Drug"):
                # Prepare input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Sex': [sex],
                    'BP': [bp],
                    'Cholesterol': [cholesterol],
                    'Na_to_K': [na_to_k]
                })
                
                # Encode categorical variables to match training data
                input_encoded = pd.get_dummies(input_data, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)
                
                # Ensure all columns from training are present (add missing with 0s)
                for col in st.session_state['feature_columns']:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                # Keep only columns used during training, in the same order
                input_encoded = input_encoded[st.session_state['feature_columns']]
                
                # Scale numerical features
                scaler = StandardScaler()
                numerical_cols = ['Age', 'Na_to_K']
                # We need to fit the scaler on the input data first
                input_encoded[numerical_cols] = scaler.fit_transform(input_encoded[numerical_cols])
                
                # Make prediction
                if model_choice == "Decision Tree":
                    prediction = st.session_state['dt_model'].predict(input_encoded)
                else:
                    prediction = st.session_state['rf_model'].predict(input_encoded)
                
                # Get drug name from prediction
                reverse_mapping = {v: k for k, v in st.session_state['drug_mapping'].items()}
                predicted_drug = reverse_mapping[prediction[0]]
                
                # Display result
                st.success(f"Recommended Drug: {predicted_drug}")
                
                # Display probabilities
                if model_choice == "Decision Tree":
                    probas = st.session_state['dt_model'].predict_proba(input_encoded)[0]
                else:
                    probas = st.session_state['rf_model'].predict_proba(input_encoded)[0]
                
                drug_names = [reverse_mapping[i] for i in range(len(probas))]
                prob_values = [float(p) for p in probas]  # Ensure probabilities are float type
                
                prob_df = pd.DataFrame({
                    'Drug': pd.Series(drug_names, dtype='string'),  # Explicitly set string type
                    'Probability': pd.Series(prob_values, dtype='float64')  # Explicitly set float type
                })
                
                # Sort by probability
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                # Plot probabilities
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x='Drug', y='Probability', hue='Drug', data=prob_df, palette='viridis', legend=False, ax=ax)
                plt.title(f"Drug Recommendation Probabilities ({model_choice})")
                plt.ylabel("Probability")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display probability table - convert to HTML to avoid PyArrow issues
                st.write("Probability for each drug class:")
                st.write(prob_df)
                
                # Additional insights
                st.subheader("Interpretation")
                st.write("""
                The model has recommended the drug based on the patient characteristics provided. 
                The probabilities show the confidence level of the model in each drug class recommendation.
                
                Key factors influencing this recommendation:
                - Na_to_K ratio (most important feature)
                - Blood Pressure status
                - Age of the patient
                """)
                
                # Allow downloading the prediction as CSV
                csv = input_data.copy()
                for col in csv.columns:
                    if csv[col].dtype == 'object':
                        csv[col] = csv[col].astype(str)

                for col in csv.columns:
                    if csv[col].dtype == 'object':
                        csv[col] = csv[col].astype('string')
                csv['Recommended_Drug'] = pd.Series([predicted_drug], dtype='string')
        
                # Generate CSV
                csv_data = csv.to_csv(index=False).encode('utf-8')
                
                # Use download button
                st.download_button(
                    label="Download Prediction Result as CSV",
                    data=csv_data,
                    file_name="drug_recommendation.csv",
                    mime="text/csv"
                )

    # Add footer
    st.markdown("---")
    st.markdown("Drug Classification App - AI5003 Assignment - © 2025 Muhammad Azhar (24K-7606)")

else:
    st.error("Dataset not loaded. Please check if the file exists and try again.")