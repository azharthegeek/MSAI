import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import base64
from io import BytesIO
import io

# Set page configuration
st.set_page_config(
    page_title="Startup Funding Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and introduction
st.title("AI5003 - Advanced Machine Learning - Startup Funding Prediction")
st.markdown("### By: Muhammad Azhar (24K-7606)")
st.markdown("### Submitted to: Professor Dr. Muhammad Rafi")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a page", 
    ["Dataset Overview", "Data Preprocessing", "Exploratory Data Analysis", 
     "Model Training & Evaluation", "Log-Transformed Models", "Make Predictions"]
)

# Function to standardize vertical categories
@st.cache_data
def standardize_vertical(vertical):
    """
    Standardizes the 'vertical' column by converting to lowercase and replacing with keywords
    """
    if not isinstance(vertical, str):
        return vertical  # Return if not string

    vertical = vertical.lower().strip()

    keywords = [
        "health", "grocery", "ecommerce", "finance", "food", "education",
        "logistics", "real estate", "consumer internet", "artificial intelligence",
        "saas", "automotive", "software", "retail", "marketing", "travel",
        "energy", "hospitality", "fitness", "gaming", "media", "hr",
        "photography", "government", "events", "sports", "fashion"
    ]

    for keyword in keywords:
        if keyword in vertical:
            vertical = keyword
            break  # Stop after the first match

    return vertical

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset for A1P1 startup_funding-dataset.csv")
        return df
    except:
        st.error("Error loading dataset. Please make sure 'dataset for A1P1 startup_funding-dataset.csv' is available.")
        return None

# Function to clean data
@st.cache_data
def clean_data(df):
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Drop unnecessary columns
    df_cleaned = df_cleaned.drop(['subvertical'], axis=1)
    
    # Drop duplicates
    df_cleaned = df_cleaned.drop_duplicates(subset=['vertical', 'investors', 'round'], keep='first')
    
    # Standardize city names
    city_mapping = {
        'Ahemadabad': 'Ahmedabad',
        'Ahemdabad': 'Ahmedabad',
        'Bengaluru': 'Bangalore',
        'Delhi': 'New Delhi',
        'Nw Delhi': 'New Delhi',
        'India/US': 'US/India',
        'USA/India': 'US/India',
        'India / US': 'US/India',
        'Kolkatta': 'Kolkata',
        'Missourie': 'Missouri',
        'Bhubneswar': 'Bhubaneswar',
        'Gurugram': 'Gurgaon'
    }
    df_cleaned['city'] = df_cleaned['city'].replace(city_mapping)
    
    # Standardize funding rounds
    round_mapping = {
        'Seed Round': 'Seed',
        'Seed Funding': 'Seed',
        'Seed funding': 'Seed',
        'Pre-series A': 'Pre-Series A',
        'pre-series A': 'Pre-Series A',
        'pre-Series A': 'Pre-Series A',
        'Venture Round': 'Venture',
        'Single Venture': 'Venture',
        'Venture - Series Unknown': 'Venture',
        'Angel Round': 'Angel',
        'Seed/ Angel Funding': 'Angel',
        'Seed / Angel Funding': 'Angel',
        'Seed/Angel Funding': 'Angel',
        'Seed / Angle Funding': 'Angel',
        'Angel / Seed Funding': 'Angel',
        'Angel Funding': 'Angel',
        'Private Equity Round': 'Private Equity',
        'PrivateEquity': 'Private Equity',
        'Private': 'Private Equity',
        'Debt Funding': 'Debt',
        'Debt and Preference capital': 'Debt',
        'Debt-Funding': 'Debt',
        'Structured Debt': 'Debt',
        'Term Loan': 'Debt',
        'Funding Round': 'Other',
        'Corporate Round': 'Other',
        'Maiden Round': 'Other',
        'Inhouse Funding': 'Other',
        'Equity': 'Other',
        'Equity Based Funding': 'Other',
        'Private Funding': 'Other',
        'Mezzanine': 'Other',
        'Series B (Extension)': 'Series B'
    }
    df_cleaned['round'] = df_cleaned['round'].replace(round_mapping)
    
    # Standardize verticals
    df_cleaned['vertical'] = df_cleaned['vertical'].apply(standardize_vertical)
    
    # Process date information
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
    df_cleaned['quarter'] = df_cleaned['date'].dt.quarter
    df_cleaned['month'] = df_cleaned['date'].dt.month
    df_cleaned.drop(columns=['date'], inplace=True)
    
    return df_cleaned

# Function to handle outliers
@st.cache_data
def handle_outliers(df):
    df_clean = df.copy()
    
    # Sort the DataFrame by 'amount' in ascending order
    df_sorted = df_clean.sort_values('amount')
    
    # Get indices to drop (10 smallest and 10 largest)
    indices_to_drop = df_sorted.index[:10].union(df_sorted.index[-10:])
    
    # Drop the rows with those indices
    df_clean = df_clean.drop(indices_to_drop)
    
    # Use IQR method for more comprehensive outlier removal
    Q1 = df_clean['amount'].quantile(0.25)
    Q3 = df_clean['amount'].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean['amount'] >= (Q1 - 1.5 * IQR)) & 
                         (df_clean['amount'] <= (Q3 + 1.5 * IQR))]
    
    # Handle zero amounts by replacing with median
    df_clean['amount'] = df_clean['amount'].replace(0, df_clean[df_clean['amount'] > 0]['amount'].median())
    
    # Create log-transformed target for better modeling
    df_clean['amount_log'] = np.log1p(df_clean['amount'])
    
    return df_clean

# Function to evaluate model with comprehensive metrics
def evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate using multiple metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'predictions': y_pred,
        'pipeline': pipeline
    }

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
            buffer = io.StringIO()
            df.info(verbose=True, buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        
        st.subheader("Dataset Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Records", df.shape[0])
        with col2:
            st.metric("Unique Startups", len(df['startup'].unique()))
        with col3:
            st.metric("Number of Features", df.shape[1])

    # Data Preprocessing page
    elif page == "Data Preprocessing":
        st.header("Data Preprocessing")
        
        # Clean data
        df_cleaned = clean_data(df)
        
        # Display steps with toggles to show results
        st.subheader("1. Removed Unnecessary Columns")
        if st.checkbox("Show details for step 1"):
            st.write("Dropped 'subvertical' column as it contains similar information to 'vertical' but with more sparsity.")
            st.text("Columns before: " + ", ".join(df.columns))
            st.text("Columns after: " + ", ".join(df_cleaned.columns))
        
        st.subheader("2. Removed Duplicate Entries")
        if st.checkbox("Show details for step 2"):
            st.write("Removed duplicates based on 'vertical', 'investors', and 'round'")
            st.metric("Records before", df.shape[0])
            st.metric("Records after", df_cleaned.shape[0])
        
        st.subheader("3. Standardized City Names")
        if st.checkbox("Show details for step 3"):
            st.write("Standardized city names to correct spelling variations and inconsistencies")
            st.write("Unique cities before:", len(df['city'].unique()))
            st.write("Unique cities after:", len(df_cleaned['city'].unique()))
            
            # Show examples of city standardization
            cities_before = pd.DataFrame(df['city'].unique(), columns=['Original Cities'])
            cities_after = pd.DataFrame(df_cleaned['city'].unique(), columns=['Standardized Cities'])
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(cities_before.head(10))
            with col2:
                st.dataframe(cities_after.head(10))
        
        st.subheader("4. Standardized Funding Rounds")
        if st.checkbox("Show details for step 4"):
            st.write("Standardized funding round names to consolidate variations")
            st.write("Unique rounds before:", len(df['round'].unique()))
            st.write("Unique rounds after:", len(df_cleaned['round'].unique()))
            
            # Show examples of round standardization
            rounds_before = pd.DataFrame(df['round'].unique(), columns=['Original Rounds'])
            rounds_after = pd.DataFrame(df_cleaned['round'].unique(), columns=['Standardized Rounds'])
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(rounds_before)
            with col2:
                st.dataframe(rounds_after)
        
        st.subheader("5. Standardized Industry Verticals")
        if st.checkbox("Show details for step 5"):
            st.write("Standardized industry verticals by mapping to common categories")
            st.write("Unique verticals before:", len(df['vertical'].unique()))
            st.write("Unique verticals after:", len(df_cleaned['vertical'].unique()))
            
            # Show examples of vertical standardization
            verticals_before = pd.DataFrame(df['vertical'].unique()[:15], columns=['Original Verticals'])
            verticals_after = pd.DataFrame(df_cleaned['vertical'].unique()[:15], columns=['Standardized Verticals'])
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(verticals_before)
            with col2:
                st.dataframe(verticals_after)
        
        st.subheader("6. Processed Date Information")
        if st.checkbox("Show details for step 6"):
            st.write("Extracted quarter and month from the date field")
            st.write("Added new features: 'quarter' and 'month'")
        
        # Handle outliers
        st.subheader("7. Handled Outliers and Created Log-transformed Target")
        df_final = handle_outliers(df_cleaned)
        if st.checkbox("Show details for step 7"):
            st.write("Removed extreme values (10 smallest and 10 largest)")
            st.write("Applied IQR method to remove outliers")
            st.write("Replaced zero amounts with the median amount")
            st.write("Created log-transformed target variable 'amount_log'")
            
            # Show amount distribution before and after
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.boxplot(x=df_cleaned["amount"], ax=ax[0], color='green')
            ax[0].set_title("Amount Distribution Before")
            
            sns.boxplot(x=df_final["amount"], ax=ax[1], color='green')
            ax[1].set_title("Amount Distribution After")
            
            st.pyplot(fig)
            
            # Show statistics before and after
            col1, col2 = st.columns(2)
            with col1:
                st.write("Statistics before outlier handling:")
                st.dataframe(df_cleaned['amount'].describe())
            with col2:
                st.write("Statistics after outlier handling:")
                st.dataframe(df_final['amount'].describe())
        
        # Save preprocessed data for subsequent pages
        st.session_state['df_final'] = df_final
        
        # Show final dataframe
        st.subheader("Final Preprocessed Dataset")
        st.dataframe(df_final.head())
        st.success(f"Preprocessing complete! Final dataset has {df_final.shape[0]} records and {df_final.shape[1]} columns.")

    # Exploratory Data Analysis page
    elif page == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        # Get preprocessed data
        if 'df_final' not in st.session_state:
            df_final = handle_outliers(clean_data(df))
            st.session_state['df_final'] = df_final
        else:
            df_final = st.session_state['df_final']
        
        # Amount Distribution Analysis
        st.subheader("Funding Amount Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=df_final["amount"], color='green', ax=ax)
            ax.set_title("Boxplot of Funding Amount")
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            tempDF = df_final[df_final['amount']>0]
            sns.histplot(tempDF['amount'], bins=30, kde=True, color='green', ax=ax)
            ax.set_xlabel("Amount")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Amount")
            st.pyplot(fig)
        
        # Log-transformed Amount
        st.subheader("Log-transformed Amount Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        tempDF = df_final[df_final['amount_log']>0]
        sns.histplot(tempDF['amount_log'], bins=30, kde=True, color='green', ax=ax)
        ax.set_xlabel("Log(Amount)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Log-Transformed Amount")
        st.pyplot(fig)
        
        # Funding by Round Type
        st.subheader("Funding by Round Type")
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(x="round", y="amount", data=df_final, showfliers=False, color='green', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Investment Amount by Funding Round")
        st.pyplot(fig)
        
        # Funding amount by top cities
        st.subheader("Funding Amount by City")
        top_cities = df_final['city'].value_counts().head(10).index
        city_data = df_final[df_final['city'].isin(top_cities)]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(x="city", y="amount", data=city_data, showfliers=False, color='green', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Investment Amount by Top Cities")
        st.pyplot(fig)
        
        # Funding by industry vertical
        st.subheader("Funding by Industry Vertical")
        top_verticals = df_final['vertical'].value_counts().head(10).index
        vertical_data = df_final[df_final['vertical'].isin(top_verticals)]
        
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.boxplot(x="vertical", y="amount", data=vertical_data, showfliers=False, color='green', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Investment Amount by Top Industry Verticals")
        st.pyplot(fig)
        
        # Funding amount by quarter
        st.subheader("Funding by Quarter")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="quarter", y="amount", data=df_final, showfliers=False, color='green', ax=ax)
        ax.set_title("Investment Amount by Quarter")
        st.pyplot(fig)
        
        # Correlation between amount and other numerical features
        st.subheader("Correlation Analysis")
        numeric_df = df_final[['amount', 'amount_log', 'quarter', 'month']]
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    # Model Training & Evaluation page
    elif page == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")
        
        # Get preprocessed data
        if 'df_final' not in st.session_state:
            df_final = handle_outliers(clean_data(df))
            st.session_state['df_final'] = df_final
        else:
            df_final = st.session_state['df_final']
        
        # Feature selection
        st.subheader("Feature Selection & Model Configuration")
        categorical_features = ['vertical', 'city', 'round']
        numerical_features = ['quarter', 'month']
        target = 'amount'
        
        # Model selection
        models_to_train = st.multiselect(
            "Select models to train",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", 
             "Random Forest", "Gradient Boosting", "SVR"],
            default=["Linear Regression", "Random Forest", "Gradient Boosting"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 1, 100, 42)
        
        # Prepare feature transformers
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])
        
        # Split data
        X = df_final[categorical_features + numerical_features]
        y = df_final[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Model dict for selected models
        model_dict = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
        }
        
        selected_models = {name: model_dict[name] for name in models_to_train}
        
        # Train models button
        if st.button("Train Models"):
            st.write(f"Training set size: {X_train.shape[0]} samples")
            st.write(f"Testing set size: {X_test.shape[0]} samples")
            
            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(selected_models.items()):
                status_text.text(f"Training {name}...")
                results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor)
                progress_bar.progress((i + 1) / len(selected_models))
            
            status_text.text("Training complete!")
            progress_bar.empty()
            
            # Save results to session state
            st.session_state['results'] = results
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['preprocessor'] = preprocessor
            
            # Show results table
            st.subheader("Model Performance Comparison")
            performance_df = pd.DataFrame({
                model: {
                    'MSE': results[model]['MSE'],
                    'RMSE': results[model]['RMSE'],
                    'MAE': results[model]['MAE'],
                    'R²': results[model]['R²']
                }
                for model in results.keys()
            }).T
            
            st.dataframe(performance_df.style.format('{:.4f}'))
            
            # Find best model
            best_model = performance_df['R²'].idxmax()
            st.success(f"Best performing model: {best_model} with R² of {performance_df.loc[best_model, 'R²']:.4f}")
            
            # Save best model info
            st.session_state['best_model'] = best_model
            
            # Cross-validation for best model
            st.subheader(f"Cross-validation for {best_model}")
            
            best_pipeline = results[best_model]['pipeline']
            cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring='r2')
            
            st.write(f"Cross-validation R² scores: {', '.join([f'{score:.4f}' for score in cv_scores])}")
            st.write(f"Mean CV R²: {np.mean(cv_scores):.4f}, Std Dev: {np.std(cv_scores):.4f}")
            
            # Visualize actual vs predicted
            st.subheader("Actual vs Predicted Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, results[best_model]['predictions'], alpha=0.7, color='green')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual Amount')
            ax.set_ylabel('Predicted Amount')
            ax.set_title(f'Actual vs Predicted Amount ({best_model})')
            st.pyplot(fig)
            
            # Plot residuals
            st.subheader("Residual Analysis")
            y_pred = results[best_model]['predictions']
            residuals = y_test - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs predicted
            ax1.scatter(y_pred, residuals, alpha=0.7, color='green')
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residual vs Predicted Values')
            
            # Histogram of residuals
            ax2.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Residuals')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature importance based on model type
            st.subheader("Feature Importance Analysis")
            
            try:
                # Get feature names after preprocessing
                cat_features = best_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_features)
                all_features = list(cat_features) + numerical_features
                
                # Extract importance based on model type
                if best_model in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                    importances = best_pipeline.named_steps['model'].feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    importance_title = f'Feature Importance ({best_model})'
                
                elif best_model in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                    coefficients = best_pipeline.named_steps['model'].coef_
                    importances = np.abs(coefficients)
                    feature_importance_df = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    importance_title = f'Feature Coefficients ({best_model})'
                
                else:
                    # For SVR or other models without direct feature importance
                    from sklearn.inspection import permutation_importance
                    
                    # Preprocess the data first
                    X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)
                    
                    # Calculate permutation importance
                    perm_importance = permutation_importance(
                        best_pipeline.named_steps['model'], X_test_transformed, y_test, n_repeats=10, random_state=random_state
                    )
                    
                    feature_importance_df = pd.DataFrame({
                        'Feature': all_features,
                        'Importance': perm_importance.importances_mean
                    }).sort_values(by='Importance', ascending=False)
                    
                    importance_title = f'Permutation Feature Importance ({best_model})'
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(12, 8))
                top_features = feature_importance_df.head(15)  # Top 15 features
                sns.barplot(x='Importance', y='Feature', data=top_features, color='green', ax=ax)
                ax.set_title(f'Top Features - {importance_title}')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.dataframe(feature_importance_df.head(15))
                
            except Exception as e:
                st.error(f"Error extracting feature importance: {str(e)}")

        # If models have been trained previously
        elif 'results' in st.session_state:
            st.info("Using previously trained models. Click 'Train Models' to retrain.")
            
            results = st.session_state['results']
            best_model = st.session_state['best_model']
            
            # Show results table
            st.subheader("Model Performance Comparison")
            performance_df = pd.DataFrame({
                model: {
                    'MSE': results[model]['MSE'],
                    'RMSE': results[model]['RMSE'],
                    'MAE': results[model]['MAE'],
                    'R²': results[model]['R²']
                }
                for model in results.keys()
            }).T
            
            st.dataframe(performance_df.style.format('{:.4f}'))
            
            # Highlight best model
                        # Highlight best model
            st.success(f"Best performing model: {best_model} with R² of {performance_df.loc[best_model, 'R²']:.4f}")
            
            # Visualize actual vs predicted
            st.subheader("Actual vs Predicted Values")
            y_test = st.session_state['y_test']
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, results[best_model]['predictions'], alpha=0.7, color='green')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Actual Amount')
            ax.set_ylabel('Predicted Amount')
            ax.set_title(f'Actual vs Predicted Amount ({best_model})')
            st.pyplot(fig)
            
            # Plot residuals
            st.subheader("Residual Analysis")
            y_pred = results[best_model]['predictions']
            residuals = y_test - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs predicted
            ax1.scatter(y_pred, residuals, alpha=0.7, color='green')
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residual vs Predicted Values')
            
            # Histogram of residuals
            ax2.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Residuals')
            
            plt.tight_layout()
            st.pyplot(fig)

    # Log-Transformed Models page
    elif page == "Log-Transformed Models":
        st.header("Log-Transformed Model Training & Evaluation")
        
        # Get preprocessed data
        if 'df_final' not in st.session_state:
            df_final = handle_outliers(clean_data(df))
            st.session_state['df_final'] = df_final
        else:
            df_final = st.session_state['df_final']
        
        # Feature selection
        st.subheader("Feature Selection & Model Configuration")
        categorical_features = ['vertical', 'city', 'round']
        numerical_features = ['quarter', 'month']
        target = 'amount_log'  # Using log-transformed target
        
        # Model selection
        models_to_train = st.multiselect(
            "Select models to train",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", 
             "Random Forest", "Gradient Boosting", "SVR"],
            default=["Linear Regression", "Random Forest", "Gradient Boosting"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 1, 100, 42)
        
        # Prepare feature transformers
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numerical_transformer, numerical_features)
            ])
        
        # Split data
        X = df_final[categorical_features + numerical_features]
        y_log = df_final[target]
        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=test_size, random_state=random_state)
        
        # Model dict for selected models
        model_dict = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
        }
        
        selected_models = {name: model_dict[name] for name in models_to_train}
        
        # Train models button
        if st.button("Train Log-transformed Models"):
            st.write(f"Training set size: {X_train.shape[0]} samples")
            st.write(f"Testing set size: {X_test.shape[0]} samples")
            
            log_results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(selected_models.items()):
                status_text.text(f"Training {name}...")
                log_results[name] = evaluate_model(model, X_train, X_test, y_train_log, y_test_log, preprocessor)
                progress_bar.progress((i + 1) / len(selected_models))
            
            status_text.text("Training complete!")
            progress_bar.empty()
            
            # Save results to session state
            st.session_state['log_results'] = log_results
            st.session_state['X_log'] = X
            st.session_state['y_log'] = y_log
            st.session_state['X_test_log'] = X_test
            st.session_state['y_test_log'] = y_test_log
            
            # Show results table
            st.subheader("Model Performance Comparison (Log-transformed target)")
            log_performance_df = pd.DataFrame({
                model: {
                    'MSE': log_results[model]['MSE'],
                    'RMSE': log_results[model]['RMSE'],
                    'MAE': log_results[model]['MAE'],
                    'R²': log_results[model]['R²']
                }
                for model in log_results.keys()
            }).T
            
            st.dataframe(log_performance_df.style.format('{:.4f}'))
            
            # Find best model
            best_log_model = log_performance_df['R²'].idxmax()
            st.success(f"Best performing model on log-transformed target: {best_log_model} with R² of {log_performance_df.loc[best_log_model, 'R²']:.4f}")
            
            # Save best model info
            st.session_state['best_log_model'] = best_log_model
            
            # Cross-validation for best model
            st.subheader(f"Cross-validation for {best_log_model}")
            
            best_log_pipeline = log_results[best_log_model]['pipeline']
            cv_scores = cross_val_score(best_log_pipeline, X, y_log, cv=5, scoring='r2')
            
            st.write(f"Cross-validation R² scores: {', '.join([f'{score:.4f}' for score in cv_scores])}")
            st.write(f"Mean CV R²: {np.mean(cv_scores):.4f}, Std Dev: {np.std(cv_scores):.4f}")
            
            # Visualize actual vs predicted (log scale)
            st.subheader("Actual vs Predicted Values (Log Scale)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test_log, log_results[best_log_model]['predictions'], alpha=0.7, color='green')
            ax.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--')
            ax.set_xlabel('Actual log(Amount)')
            ax.set_ylabel('Predicted log(Amount)')
            ax.set_title(f'Actual vs Predicted log(Amount) ({best_log_model})')
            st.pyplot(fig)
            
            # Plot residuals (log scale)
            st.subheader("Residual Analysis (Log Scale)")
            log_y_pred = log_results[best_log_model]['predictions']
            log_residuals = y_test_log - log_y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs predicted (log scale)
            ax1.scatter(log_y_pred, log_residuals, alpha=0.7, color='green')
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted Log Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residual vs Predicted Values (Log Scale)')
            
            # Histogram of residuals (log scale)
            ax2.hist(log_residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Residuals (Log Scale)')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Convert log predictions back to original scale and evaluate
            st.subheader("Performance in Original Scale")
            y_test_orig = np.expm1(y_test_log)
            y_pred_orig = np.expm1(log_results[best_log_model]['predictions'])
            
            # Calculate metrics in original scale
            mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
            rmse_orig = np.sqrt(mse_orig)
            mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
            r2_orig = r2_score(y_test_orig, y_pred_orig)
            
            # Display metrics in original scale
            metrics_orig = {
                'MSE': mse_orig,
                'RMSE': rmse_orig,
                'MAE': mae_orig,
                'R²': r2_orig
            }
            
            st.write("Performance of best log model in original scale:")
            st.dataframe(pd.DataFrame([metrics_orig]).T.rename(columns={0: 'Value'}).style.format('{:.4f}'))
            
            # Compare with best non-log model if available
            if 'best_model' in st.session_state and 'results' in st.session_state:
                st.subheader("Comparison with Non-Log Model")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Log-Transformed Model R²", f"{r2_orig:.4f}")
                    st.write(f"Best model: {best_log_model}")
                    
                with col2:
                    best_non_log = st.session_state['best_model']
                    non_log_r2 = st.session_state['results'][best_non_log]['R²']
                    st.metric("Original Scale Model R²", f"{non_log_r2:.4f}")
                    st.write(f"Best model: {best_non_log}")
                
                st.write("""
                **Interpretation:**
                
                The log-transformed approach typically performs better when data has:
                - Positive skewness (which is common in financial data like funding amounts)
                - Multiplicative error structures
                - Non-constant variance (heteroscedasticity)
                
                When the R² in the original scale is higher for the log-transformed model, it indicates that the 
                log transformation is appropriate for this data.
                """)
        
        # If models have been trained previously
        elif 'log_results' in st.session_state:
            st.info("Using previously trained log-transformed models. Click 'Train Log-transformed Models' to retrain.")
            
            log_results = st.session_state['log_results']
            best_log_model = st.session_state['best_log_model']
            y_test_log = st.session_state['y_test_log']
            
            # Show results table
            st.subheader("Model Performance Comparison (Log-transformed target)")
            log_performance_df = pd.DataFrame({
                model: {
                    'MSE': log_results[model]['MSE'],
                    'RMSE': log_results[model]['RMSE'],
                    'MAE': log_results[model]['MAE'],
                    'R²': log_results[model]['R²']
                }
                for model in log_results.keys()
            }).T
            
            st.dataframe(log_performance_df.style.format('{:.4f}'))
            
            # Highlight best model
            st.success(f"Best performing model on log-transformed target: {best_log_model} with R² of {log_performance_df.loc[best_log_model, 'R²']:.4f}")
            
            # Visualize actual vs predicted (log scale)
            st.subheader("Actual vs Predicted Values (Log Scale)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test_log, log_results[best_log_model]['predictions'], alpha=0.7, color='green')
            ax.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--')
            ax.set_xlabel('Actual log(Amount)')
            ax.set_ylabel('Predicted log(Amount)')
            ax.set_title(f'Actual vs Predicted log(Amount) ({best_log_model})')
            st.pyplot(fig)

    # Make Predictions page
    elif page == "Make Predictions":
        st.header("Make Predictions for New Startups")
        
        # Check if models have been trained
        if 'results' not in st.session_state or 'log_results' not in st.session_state:
            st.warning("Please train both standard and log-transformed models first.")
        else:
            st.subheader("Enter Startup Information")
            
            # Get list of unique values for categorical features
            if 'df_final' not in st.session_state:
                df_final = handle_outliers(clean_data(df))
                st.session_state['df_final'] = df_final
            else:
                df_final = st.session_state['df_final']
            
            # Get top 10 most common values for each categorical feature
            top_cities = df_final['city'].value_counts().head(10).index.tolist()
            top_verticals = df_final['vertical'].value_counts().head(10).index.tolist()
            funding_rounds = df_final['round'].unique().tolist()
            
            # Create form for input
            col1, col2 = st.columns(2)
            
            with col1:
                city = st.selectbox("City", options=[""] + top_cities + ["Other"])
                if city == "Other":
                    city = st.text_input("Enter city name:")
                
                vertical = st.selectbox("Industry Vertical", options=[""] + top_verticals + ["Other"])
                if vertical == "Other":
                    vertical = st.text_input("Enter industry vertical:")
                    vertical = standardize_vertical(vertical)
                
                funding_round = st.selectbox("Funding Round", options=[""] + funding_rounds)
                
            with col2:
                quarter = st.selectbox("Quarter", options=[1, 2, 3, 4])
                month = st.slider("Month", 1, 12, 6)
            
            # Model selection
            model_type = st.radio(
                "Choose prediction approach",
                ["Standard Model", "Log-Transformed Model (Recommended)"]
            )
            
            if st.button("Predict Funding Amount"):
                # Validate input
                if not (city and vertical and funding_round):
                    st.error("Please fill in all fields before prediction.")
                else:
                    # Prepare input data
                    input_data = pd.DataFrame({
                        'city': [city],
                        'vertical': [vertical],
                        'round': [funding_round],
                        'quarter': [quarter],
                        'month': [month]
                    })
                    
                    # Get the best model based on selection
                    if model_type == "Standard Model":
                        if 'best_model' not in st.session_state:
                            st.error("Standard model not trained yet.")
                            st.stop()
                        
                        best_model = st.session_state['best_model']
                        pipeline = st.session_state['results'][best_model]['pipeline']
                        
                        # Make prediction
                        prediction = pipeline.predict(input_data)[0]
                        
                        # Display result
                        st.success(f"Predicted Funding Amount: ${prediction:,.2f}")
                        
                    else:  # Log-Transformed Model
                        if 'best_log_model' not in st.session_state:
                            st.error("Log-transformed model not trained yet.")
                            st.stop()
                        
                        best_model = st.session_state['best_log_model']
                        pipeline = st.session_state['log_results'][best_model]['pipeline']
                        
                        # Make log-scale prediction and transform back
                        log_prediction = pipeline.predict(input_data)[0]
                        prediction = np.expm1(log_prediction)
                        
                        # Display result
                        st.success(f"Predicted Funding Amount: ${prediction:,.2f}")
                    
                    # Show similar startups for reference
                    st.subheader("Similar Startups for Reference")
                    
                    # Find similar startups based on vertical and round
                    similar = df_final[
                        (df_final['vertical'] == vertical) & 
                        (df_final['round'] == funding_round)
                    ].sort_values('amount', ascending=False)
                    
                    if len(similar) > 0:
                        # Show statistics
                        stats = similar['amount'].describe()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Funding", f"${stats['mean']:,.2f}")
                        with col2:
                            st.metric("Median Funding", f"${stats['50%']:,.2f}")
                        with col3:
                            st.metric("Max Funding", f"${stats['max']:,.2f}")
                        
                        # Show similar startups
                        st.write("Recent startups in the same vertical and funding round:")
                        
                        # Select columns to display
                        display_cols = ['startup', 'city', 'vertical', 'round', 'amount', 'investors']
                        st.dataframe(similar[display_cols].head(5))
                        
                        # Plot distribution of similar startups
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.histplot(similar['amount'], bins=15, kde=True, color='green', ax=ax)
                        ax.axvline(x=prediction, color='red', linestyle='--', linewidth=2)
                        ax.text(prediction, ax.get_ylim()[1]*0.9, 'Prediction', rotation=90, color='red')
                        ax.set_xlabel('Funding Amount')
                        ax.set_ylabel('Number of Startups')
                        ax.set_title('Distribution of Funding for Similar Startups')
                        st.pyplot(fig)
                        
                    else:
                        st.info("No similar startups found with the same vertical and funding round.")

    # Add footer
    st.markdown("---")
    st.markdown("Startup Funding Prediction App - AI5003 Assignment - © 2025 Muhammad Azhar (24K-7606)")

else:
    st.error("Dataset not loaded. Please check if the file exists and try again.")