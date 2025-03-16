# -*- coding: utf-8 -*-
"""ML Assignment 01 Problem 01 Regression 24K-7606

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t8r7M80dYdE8R5cm3HiKheue5NRwYoxj

## AI5003 - Advance Machine Learning - Assignment No. 1 - Problem 01 - (Regression)
## Name: **Muhammad Azhar**
## ID: **24K-7606**
## Submitted to: **Professor Dr. Muhammad Rafi**

Data Loading And Initial Exploration
"""

!pip install ydata-profiling -q

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

df = pd.read_csv("dataset for A1P1 startup_funding-dataset.csv") # Load the dataset

# Generate a comprehensive profile report
profile = ProfileReport(df)
profile.to_file(output_file="startup-data-analytics.html")

df.info() # Display basic information about the dataset

df.head() # Display the first few rows of the dataset

"""Data cleaning and preprocessing"""

df.isnull().sum() # Check for missing values

len(df['startup'].unique()) # Check unique startups to understand dataset size

df = df.drop(['subvertical'],axis=1) # Drop unnecessary columns

df[df.duplicated(subset=['startup','vertical', 'investors', 'round'], keep=False)] # Check and remove duplicates

df = df.drop_duplicates(subset=['vertical', 'investors', 'round'], keep='first')

# Standardize city names
df['city'] = df['city'].replace({
    'Ahemadabad': 'Ahmedabad',
    'Ahemdabad': 'Ahmedabad',
    'Bengaluru': 'Bangalore',
    'Delhi': 'New Delhi',
    'Nw Delhi': 'New Delhi',
    'India/US': 'US/India',
    'USA/India': 'US/India',
    'India / US': 'US/India',
    'Kolkatta':'Kolkata',
    'Missourie': 'Missouri',
    'Bhubneswar': 'Bhubaneswar',
    'Gurugram': 'Gurgaon'

})

df['city'].unique()

# Standardize funding rounds
df['round'] = df['round'].replace({
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
})

len(df['round'].unique()) #Number of unique funding rounds after standardization

df['round'].unique()

len(df['vertical'].unique()) # Check and standardize industry verticals

#Prompt: Create a funtion to standarize the vertical search for a keyword in a vertical and if the keyword math any word then replace it with the keyword

def standardize_vertical(vertical):
    """
    Standardizes the 'vertical' column by:
    1. Converting to lowercase and removing spaces.
    2. Auto-detecting and replacing keywords.
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

df['vertical'] = df['vertical'].apply(standardize_vertical)

len(df['vertical'].unique()) #Number of unique industry verticals after standardization

df['investors'].unique()

len(df['investors'].unique())

df["date"][1]

#Processing date information
df['date'] = pd.to_datetime(df['date'])
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df.drop(columns=['date'], inplace=True)

"""Outlier detection and handling"""

df['amount'].describe()

#Prompt drop 10 minimum and maximum value from df['amount']
#Method 1: Remove extreme values (smallest and largest)
# Sort the DataFrame by 'amount' in ascending order
df_sorted = df.sort_values('amount')

# Get indices to drop (10 smallest and 10 largest)
indices_to_drop = df_sorted.index[:10].union(df_sorted.index[-10:])

# Drop the rows with those indices
df = df.drop(indices_to_drop)

# Method 2: Use IQR method for more comprehensive outlier removal
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['amount'] >= (Q1 - 1.5 * IQR)) & (df['amount'] <= (Q3 + 1.5 * IQR))]

df['amount'] = df['amount'].replace(0, df[df['amount'] > 0]['amount'].median()) # Handle zero amounts by replacing with median
df['amount_log'] = np.log1p(df['amount']) # Create log-transformed target for better modeling

"""Exploratory Data Analysis and Visualization"""

# Visualize amount distribution with boxplot
plt.figure(figsize=(5, 5))
sns.boxplot(x=df["amount"], color='green')
plt.title("Boxplot of Funding Amount")
plt.show()

# Histogram of amount
tempDF = df[df['amount']>0]
plt.figure(figsize=(8, 6))
sns.histplot(tempDF['amount'], bins=30, kde=True, color='green')
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Amount")
plt.show()

# Histogram of log-transformed amount
tempDF = df[df['amount_log']>0]
plt.figure(figsize=(8, 6))
sns.histplot(tempDF['amount_log'], bins=30, kde=True, color='green')
plt.xlabel("Log(Amount)")
plt.ylabel("Frequency")
plt.title("Distribution of Log-Transformed Amount")
plt.show()

# Funding amount by round type
plt.figure(figsize=(15,8))
sns.boxplot(x="round", y="amount", data=df, showfliers=False,color='green')
plt.xticks(rotation=45)
plt.title("Investment Amount by Funding Round")
plt.show()

df.head(10)

"""Model Building And Evaluation"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature selection and preprocessing
categorical_features = ['vertical', 'city', 'round']
numerical_features = ['quarter', 'month']
target = 'amount'

# Prepare feature transformers
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine transformers into a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Split data into features and target
X = df[categorical_features + numerical_features]
y = df[target]

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

#Prompt: Apply Linear,Ridge,Lasso, Decision tree , Random Forest, Gradient Boosting and SVR on this startup funding dataset
# Initialize multiple regression models for comparison
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
}

# Function to evaluate models with comprehensive metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
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

"""Model training and evaluation on original scale"""

results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"MSE: {results[name]['MSE']:.2f}")
    print(f"RMSE: {results[name]['RMSE']:.2f}")
    print(f"MAE: {results[name]['MAE']:.2f}")
    print(f"R²: {results[name]['R²']:.2f}")

# Compare model performances
print("\nModel Performance Comparison:")
performance_df = pd.DataFrame({
    model: {
        'MSE': results[model]['MSE'],
        'RMSE': results[model]['RMSE'],
        'MAE': results[model]['MAE'],
        'R²': results[model]['R²']
    }
    for model in models.keys()
}).T

print(performance_df)

# Find the best model based on R²
best_model = performance_df['R²'].idxmax()
print(f"\nBest performing model: {best_model} with R² of {performance_df.loc[best_model, 'R²']:.2f}")

# Cross-validation for the best model
best_pipeline = results[best_model]['pipeline']
cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring='r2')
print(f"\nCross-validation R² scores for {best_model}: {cv_scores}")
print(f"Mean CV R²: {np.mean(cv_scores):.2f}, Std Dev: {np.std(cv_scores):.2f}")

"""Model visualization and interpretation"""

# Visualize actual vs predicted values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, results[best_model]['predictions'], alpha=0.7,color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', )
plt.xlabel('Actual Amount')
plt.ylabel('Predicted Amount')
plt.title(f'Actual vs Predicted Amount ({best_model})')
plt.show()

# Plot residuals to analyze model performance (THIS WAS MISSING in original code)
y_pred = results[best_model]['predictions']
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7,color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'Residual Plot for {best_model}')
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title(f'Distribution of Residuals for {best_model}')
plt.show()

# Get feature importance regardless of model type
print("\nExtracting feature importance information...")

# First get feature names after preprocessing
cat_features = best_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_features)
all_features = list(cat_features) + numerical_features

# Get feature importance based on model type
if best_model in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
    # Direct feature importance for tree-based models
    importances = best_pipeline.named_steps['model'].feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    importance_title = f'Feature Importance ({best_model})'

elif best_model in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    # For linear models, use coefficients as importance
    coefficients = best_pipeline.named_steps['model'].coef_
    # Take absolute values for importance ranking
    importances = np.abs(coefficients)
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    importance_title = f'Feature Coefficients ({best_model})'

else:
    # For SVR or other models without direct feature importance
    # Use permutation importance
    from sklearn.inspection import permutation_importance

    # Preprocess the data first
    X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)

    # Calculate permutation importance
    perm_importance = permutation_importance(
        best_pipeline.named_steps['model'], X_test_transformed, y_test, n_repeats=10, random_state=42
    )

    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    importance_title = f'Permutation Feature Importance ({best_model})'

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10) , color='green')
plt.title(f'Top 10 {importance_title}')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

"""Log-transformed model training and evaluation"""

# Define target as logarithmic amount
y_log = df['amount_log']

# Split data for log-transformed target
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Evaluate models with log-transformed target
log_results = {}
for name, model in models.items():
    print(f"\nEvaluating {name} on log-transformed target...")
    log_results[name] = evaluate_model(model, X_train_log, X_test_log, y_train_log, y_test_log)
    print(f"MSE: {log_results[name]['MSE']:.2f}")
    print(f"RMSE: {log_results[name]['RMSE']:.2f}")
    print(f"MAE: {log_results[name]['MAE']:.2f}")
    print(f"R²: {log_results[name]['R²']:.2f}")

# Compare model performances on log-transformed target
log_performance_df = pd.DataFrame({
    model: {
        'MSE': log_results[model]['MSE'],
        'RMSE': log_results[model]['RMSE'],
        'MAE': log_results[model]['MAE'],
        'R²': log_results[model]['R²']
    }
    for model in models.keys()
}).T

print("\nModel Performance Comparison (Log-transformed target):")
print(log_performance_df)

# Find the best model based on R² for log-transformed target
best_log_model = log_performance_df['R²'].idxmax()
print(f"\nBest performing model on log-transformed target: {best_log_model} with R² of {log_performance_df.loc[best_log_model, 'R²']:.2f}")

"""Visualization and back-transformation"""

# Visualize actual vs predicted log values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_log, log_results[best_log_model]['predictions'], alpha=0.7,color='green')
plt.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--')
plt.xlabel('Actual log(Amount)')
plt.ylabel('Predicted log(Amount)')
plt.title(f'Actual vs Predicted log(Amount) ({best_log_model})')
plt.savefig('actual_vs_predicted_log.png')
plt.show()

# Plot residuals for log model
log_y_pred = log_results[best_log_model]['predictions']
log_residuals = y_test_log - log_y_pred

plt.figure(figsize=(10, 6))
plt.scatter(log_y_pred, log_residuals, alpha=0.7,color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Log Values')
plt.ylabel('Residuals')
plt.title(f'Residual Plot for {best_log_model} (Log Scale)')
plt.savefig('log_residual_plot.png')
plt.show()

# Convert log predictions back to original scale and evaluate
y_test_orig = np.expm1(y_test_log)
y_pred_orig = np.expm1(log_results[best_log_model]['predictions'])

# Calculate metrics in original scale
mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
rmse_orig = np.sqrt(mse_orig)
mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
r2_orig = r2_score(y_test_orig, y_pred_orig)

print("\nPerformance of best log model in original scale:")
print(f"MSE: {mse_orig:.2f}")
print(f"RMSE: {rmse_orig:.2f}")
print(f"MAE: {mae_orig:.2f}")
print(f"R²: {r2_orig:.2f}")

"""Conclusion and Model Selection"""

print(f"\n1. For predicting the exact amount, the best model is: {best_model}")
print(f"   with R² of {performance_df.loc[best_model, 'R²']:.4f}")

print(f"\n2. For predicting the logarithm of the amount (which is more suitable for this data), ")
print(f"   the best model is: {best_log_model}")
print(f"   with R² of {log_performance_df.loc[best_log_model, 'R²']:.4f}")

#Prompt Help me write report on the above results

"""# Startup Funding Prediction Model: Interpretation and Reporting

## Key Findings

1. **Model Performance**:
   - Original scale: Lasso Regression performed best (R² = 0.32)
   - Log-transformed scale: Gradient Boosting performed best (R² = 0.39)
   - Log transformation improved model performance, suggesting funding amounts follow a log-normal distribution

2. **Most Influential Predictors**:
   - **Funding Round Type**: The strongest predictor of funding amount (Seed rounds: 19.38%, Angel rounds: 16.24%)
   - **Industry Vertical**: Finance (6.02%) and logistics (5.47%) sectors attract significantly higher funding
   - **Location**: Startups in Bangalore (2.31%) receive higher funding amounts
   - **Timing**: The quarter of funding has minimal impact (0.18%)

3. **Model Stability**:
   - Cross-validation shows moderate variability (R² from 0.08 to 0.33)
   - Mean CV R² of 0.22 indicates reasonable but not exceptional predictive power

## Conclusion

The analysis reveals that startup funding amounts are moderately predictable (explaining ~39% of variance) when using a log-transformed approach. The funding round type is the dominant factor in determining investment size, with seed and angel rounds showing distinct patterns. Industry vertical also plays a significant role, with finance and logistics attracting higher investments.

For practical applications, the Gradient Boosting model with log-transformation is recommended for predicting funding amounts. The model's performance suggests that while certain factors can be quantified, substantial variability in startup funding remains unexplained by the available features, likely due to qualitative factors such as founding team experience, market conditions, and investor relationships not captured in the dataset.
"""