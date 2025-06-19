import pandas as pd
import numpy as np
import re
import string
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("glassdoor_jobs.csv")
df = df[df['Salary Estimate'] != '-1']
df.dropna(subset=['Job Title', 'Salary Estimate'], inplace=True)

# Clean salary
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided' in x.lower() else 0)
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
salary = salary.str.replace('K', '').str.replace('$', '')
salary = salary.str.lower().str.replace('per hour', '').str.replace('employer provided salary:', '')
df['min_salary'] = salary.apply(lambda x: int(x.split('-')[0].strip()))
df['max_salary'] = salary.apply(lambda x: int(x.split('-')[1].strip()))
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# Simplify job title
def simplify_title(title):
    title = title.lower()
    if 'data scientist' in title:
        return 'data scientist'
    elif 'data engineer' in title:
        return 'data engineer'
    elif 'analyst' in title:
        return 'analyst'
    elif 'machine learning' in title:
        return 'mle'
    elif 'manager' in title:
        return 'manager'
    elif 'director' in title:
        return 'director'
    elif 'software engineer' in title:
        return 'software engineer'
    else:
        return 'other'

df['job_simplified'] = df['Job Title'].apply(simplify_title)
df['python'] = df['Job Description'].str.lower().str.contains('python').astype(int)
df['excel'] = df['Job Description'].str.lower().str.contains('excel').astype(int)
df['aws'] = df['Job Description'].str.lower().str.contains('aws').astype(int)
df['sql'] = df['Job Description'].str.lower().str.contains('sql').astype(int)

# Remove outliers
Q1 = df['avg_salary'].quantile(0.25)
Q3 = df['avg_salary'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['avg_salary'] >= Q1 - 1.5 * IQR) & (df['avg_salary'] <= Q3 + 1.5 * IQR)]

# One-hot encoding
columns_to_encode = [col for col in ['job_simplified', 'Location', 'Type of ownership'] if col in df.columns]
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# Feature Selection
X = df.drop(['avg_salary', 'avg_salary_inr'], axis=1, errors='ignore')
X = pd.get_dummies(X, drop_first=True)
y = df['avg_salary']

selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[selected_features])

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Save model and transformers
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "features": selected_features.tolist()
    }, f)

print("Model saved as model.pkl")
