# 💼 Glassdoor Salary Predictor
A Machine Learning-powered web application that predicts the average salary for job roles using features like job type, skills, and salary range. Built using Python, Streamlit, and XGBoost.

🔍 Overview
This project uses data from Glassdoor job postings to analyze and predict the salary range for various job profiles. The model has been trained using important job-related attributes and wrapped inside an interactive Streamlit UI.

Users can input:

Technical skills (Python, SQL, AWS, Excel)

Job type (Hourly/Employer-Provided)

Salary range (Minimum and Maximum)

Encoded job and company details

The app then predicts the average salary in USD and INR.

🚀 Features
 Data Cleaning & EDA using pandas and seaborn

 ML Model: XGBoost Regressor with PCA

 Feature Engineering: Skill parsing, job title simplification, and outlier handling

 Hyperparameter tuning using RandomizedSearchCV

 Web App UI built with Streamlit

 Model serialized using pickle

🛠 Tech Stack
Tool	& Purpose
Python	Programming language
pandas / numpy	Data manipulation
seaborn / matplotlib	Data visualization
scikit-learn	Preprocessing, metrics, tuning
XGBoost	Regressor model
PCA	Dimensionality reduction
Streamlit	Web UI
Pickle	Model serialization

🧪 Sample Input
Knows Python ✅

Not hourly ❌

Min Salary = 85 (K)

Max Salary = 120 (K)

💡 Output:

Predicted Salary: $102,000
Salary in INR: ₹8,46,600

Feel free to connect or suggest improvements!
🌐 https://www.linkedin.com/in/khushi-sonare2003/


