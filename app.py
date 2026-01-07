import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

print("\n--- Customer Purchase Prediction ---\n")

age = int(input("Enter Age: "))
gender = int(input("Gender (Male=1, Female=0): "))
income = int(input("Annual Income: "))
score = int(input("Spending Score: "))
prev = int(input("Previous Purchase (Yes=1, No=0): "))
eng = int(input("Engagement Score: "))

data = np.array([[age, gender, income, score, prev, eng]])
data = scaler.transform(data)

prediction = model.predict(data)
probability = model.predict_proba(data)[0][1]

if prediction[0] == 1:
    print(f"\n✅ Customer WILL Purchase (Probability: {probability*100:.2f}%)")
else:
    print(f"\n❌ Customer will NOT Purchase (Probability: {probability*100:.2f}%)")
