🛒 Customer Purchase Prediction Using Classification Algorithms
📌 Project Overview

This project builds an end-to-end Machine Learning classification system that predicts whether a customer will purchase a product or service based on demographic and behavioral data.

The project demonstrates a complete ML workflow including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and real-time prediction using a saved model.

This project is suitable for Beginner to Intermediate level learners and is internship / resume ready.

🎯 Objective

To develop a classification model that predicts customer purchase behavior using:

Demographic information

Past purchase history

Engagement metrics

The final system allows user input and displays prediction output with probability.

🧠 Machine Learning Concepts Used

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Encoding & Scaling

Classification Algorithms

Model Evaluation Metrics

Model Serialization (Pickle)

Real-time Prediction

📂 Project Structure
Customer_Purchase_Prediction/
│
├── data/
│   └── customer_data.csv
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── app.py
├── requirements.txt
└── README.md

📊 Dataset Description

The dataset contains the following features:

Feature Name	Description
Age	Age of the customer
Gender	Male / Female
Annual_Income	Customer’s yearly income
Spending_Score	Spending behavior score
Previous_Purchase	Past purchase history
Engagement_Score	Engagement level
Purchased	Target Variable (0 = No, 1 = Yes)
⚙️ Project Workflow
1️⃣ Data Loading & Understanding

Load dataset using Pandas

Inspect shape, columns, and data types

Identify target variable

2️⃣ Data Preprocessing

Handle missing values

Encode categorical variables using Label Encoding

Scale numerical features using StandardScaler

Split data into training and testing sets

3️⃣ Exploratory Data Analysis (EDA)

Feature distributions

Correlation heatmap

Customer behavior insights

4️⃣ Model Building

The following classification models were implemented:

Logistic Regression

Random Forest Classifier

5️⃣ Model Evaluation

Models were evaluated using:

Accuracy

Confusion Matrix

Classification Report

The Random Forest model achieved the highest accuracy and was selected as the final model.

6️⃣ Model Deployment

Trained model saved using pickle

Scaler saved for consistent preprocessing

app.py loads model and performs predictions based on user input

🧪 Model Performance (Sample)
Model	Accuracy
Logistic Regression	~78%
Random Forest	~90% ✅
🔢 Input & Output Example
🔹 Input
Age: 27
Gender: Male
Annual Income: 45000
Spending Score: 62
Previous Purchase: Yes
Engagement Score: 7

🔹 Output
✅ Customer WILL Purchase
Probability: 86%

🚀 How to Run the Project
Step 1: Clone or Download Project
git clone <repository-url>
cd Customer_Purchase_Prediction

Step 2: Create & Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Train the Model

Open notebooks/EDA_and_Modeling.ipynb

Run all cells

This will create:

models/best_model.pkl

models/scaler.pkl

Step 5: Run the Application
python app.py

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Pickle

VS Code

Jupyter Notebook

📈 Learning Outcomes

By completing this project, you will:

Understand classification algorithms

Learn data preprocessing techniques

Analyze customer behavior

Evaluate ML models effectively

Deploy ML models for real-time prediction

🔮 Future Enhancements

Hyperparameter tuning using GridSearchCV

Convert into a Streamlit Web App

Deploy using Flask or FastAPI

Use a larger real-world dataset

Add database connectivity

👨‍🎓 Author

Jangeti Saikirit
B.Tech – Computer Science & Engineering (CSM)
Machine Learning Enthusiast

⭐ Acknowledgments

Scikit-learn Documentation

Kaggle Datasets

Open-source ML Community