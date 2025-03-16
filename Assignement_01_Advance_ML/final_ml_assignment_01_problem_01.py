# -*- coding: utf-8 -*-
"""ML Assignment 01 Problem 01
"""

!pip install ydata-profiling -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

df = pd.read_csv("dataset for A1P1 startup_funding-dataset.csv")

profile = ProfileReport(df)
profile.to_file(output_file="startup-data-analytics.html")

df.info()

df.head()

df.isnull().sum()

len(df['startup'].unique())

# df = df.drop(['subvertical','startup'],axis=1)
df = df.drop(['subvertical'],axis=1)

df[df.duplicated(subset=['startup','vertical', 'investors', 'round'], keep=False)]

df = df.drop_duplicates(subset=['vertical', 'investors', 'round'], keep='first')

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

df['round'].unique()

len(df['vertical'].unique())

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
    "health",
    "grocery",
    "ecommerce",
    "finance",
    "food",
    "education",
    "logistics",
    "real estate",
    "consumer internet",
    "artificial intelligence",
    "saas",
    "automotive",
    "software",
    "retail",
    "marketing",
    "travel",
    "energy",
    "hospitality",
    "fitness",
    "gaming",
    "media",
    "hr",
    "photography",
    "government",
    "events",
    "sports",
    "fashion"
]
    for keyword in keywords:
        if keyword in vertical:
            vertical = keyword
            break  # Stop after the first match

    return vertical

df['vertical'] = df['vertical'].apply(standardize_vertical)

len(df['vertical'].unique())

df['investors'].unique()

len(df['investors'].unique())

df["date"][1]

df['date'] = pd.to_datetime(df['date'])
df['quarter'] = df['date'].dt.quarter
# df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

df.drop(columns=['date'], inplace=True)

df['amount'].describe()


# Sort the DataFrame by 'amount' in ascending order
df_sorted = df.sort_values('amount')

# Get indices to drop (10 smallest and 10 largest)
indices_to_drop = df_sorted.index[:10].union(df_sorted.index[-10:])

# Drop the rows with those indices
df = df.drop(indices_to_drop)

Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['amount'] >= (Q1 - 1.5 * IQR)) & (df['amount'] <= (Q3 + 1.5 * IQR))]

df['amount'] = df['amount'].replace(0, df[df['amount'] > 0]['amount'].median())
df['amount_log'] = np.log1p(df['amount'])

plt.figure(figsize=(5, 5))
sns.boxplot(x=df["amount"])
plt.show()

tempDF = df[df['amount']>0]
plt.figure(figsize=(8, 6))
sns.histplot(tempDF['amount'], bins=30, kde=True)
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Amount")
plt.show()

tempDF = df[df['amount_log']>0]
plt.figure(figsize=(8, 6))
sns.histplot(tempDF['amount_log'], bins=30, kde=True)
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Amount")
plt.show()

df.head(10)

plt.figure(figsize=(15,8))
sns.boxplot(x="round", y="amount", data=df, showfliers=False)
plt.xticks(rotation=45)
plt.title("Investment Amount by Funding Round")
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Feature selection and preprocessing
categorical_features = ['vertical', 'city', 'round']
numerical_features = [ 'quarter','month']
target = 'amount'

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

# Split data into features and target
X = df[categorical_features + numerical_features]
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize regression models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate
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

# Evaluate each model and store results
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

# Visualize actual vs predicted values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, results[best_model]['predictions'], alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Amount')
plt.ylabel('Predicted Amount')
plt.title(f'Actual vs Predicted Amount ({best_model})')
plt.savefig('actual_vs_predicted.png')

# Feature importance for tree-based models
if best_model in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
    # Get feature names after one-hot encoding
    cat_features = best_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = list(cat_features) + numerical_features

    # Extract feature importances
    importances = best_pipeline.named_steps['model'].feature_importances_

    # Create DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance_df.head(10))

    # Plot feature importances
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title(f'Top 10 Feature Importances ({best_model})')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

# Try different models on logarithmic target
print("\n\nTraining models on logarithmic target (amount_log):")

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

# Visualize actual vs predicted log values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test_log, log_results[best_log_model]['predictions'], alpha=0.7)
plt.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--')
plt.xlabel('Actual log(Amount)')
plt.ylabel('Predicted log(Amount)')
plt.title(f'Actual vs Predicted log(Amount) ({best_log_model})')
plt.savefig('actual_vs_predicted_log.png')

# Convert log predictions back to original scale
y_test_orig = np.expm1(y_test_log) #Use expm1 to reverse log1p
y_pred_orig = np.expm1(log_results[best_log_model]['predictions'])#Use expm1 to reverse log1p

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

print("\nConclusion:")
print(f"For predicting the exact amount, the best model is: {best_model}")
print(f"For predicting the logarithm of the amount (which may be more suitable for this data), the best model is: {best_log_model}")

