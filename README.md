**Project Description**

This project is a Machine Learning–based web application that predicts the risk of heart disease using patient health information.
The system allows users to train a model, evaluate its performance, and predict heart disease risk through an interactive web interface.
The project demonstrates the end-to-end ML workflow, including data preprocessing, model training, evaluation, API development, and frontend integration.

**Problem Addressed**

Heart disease is one of the leading causes of death worldwide.
Early prediction helps in preventive care and medical decision-making.

**This system aims to assist in:**

Identifying high-risk patients

Supporting early diagnosis

Demonstrating real-world ML deployment

**Dataset Used**
https://www.kaggle.com/datasets/rashadrmammadov/heart-disease-prediction

**The dataset contains patient medical and lifestyle information such as:**

Age
Gender
Cholesterol
Blood Pressure
Heart Rate
Blood Sugar
Smoking Status
Alcohol Intake
Exercise Hours
Stress Level
Family History
Diabetes
Obesity
Chest Pain Type
Exercise Induced Angina
**Target Variable:**

1 → High Risk of Heart Disease
0 → Low Risk of Heart Disease

**Technologies Used**

Python

Scikit-learn – Machine Learning

Pandas & NumPy – Data handling

FastAPI – Backend REST APIs

Streamlit – Frontend web interface

Joblib – Model serialization

Machine Learning Model

Random Forest Classifier

Handles both numerical and categorical features

Provides prediction probability (confidence score)

**Features**

Upload dataset for model training

Upload dataset for model evaluation

Displays accuracy and classification report

Patient data input through UI

**Predicts:**

High Risk or Low Risk

Prediction confidence percentage

Clean, centered, and user-friendly interface

Modular and extendable architecture

**Application Workflow**

Upload training dataset

Train the ML model

Upload testing dataset

Evaluate model performance

Enter patient details

Get heart disease risk prediction

**Output**

**The application displays:**

Heart Disease Risk (High / Low)

Confidence Score (%)
