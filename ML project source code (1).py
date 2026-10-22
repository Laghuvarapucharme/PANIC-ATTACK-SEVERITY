import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# ----------------------------------------------------
# USE YOUR CORRECT FILE PATH
# ----------------------------------------------------
file_path = r"C:\Users\yases\OneDrive\Documents\ML project\panic_attack_dataset for ml project.csv"

# Check file exists first
if not os.path.exists(file_path):
    raise FileNotFoundError(f"\n❌ File not found at:\n{file_path}\nCheck spelling or spaces.")

# Load dataset
df = pd.read_csv(file_path)

# Features
features = ['Age', 'Gender', 'Panic_Attack_Frequency', 'Duration_Minutes', 'Trigger',
            'Heart_Rate', 'Sweating', 'Shortness_of_Breath', 'Dizziness', 'Chest_Pain',
            'Trembling', 'Medical_History', 'Medication', 'Caffeine_Intake',
            'Exercise_Frequency', 'Sleep_Hours', 'Alcohol_Consumption', 'Smoking', 'Therapy']

df = df[features + ['Panic_Score']]

# Convert Yes/No columns
binary_columns = ['Smoking', 'Therapy', 'Medication', 'Sweating',
                  'Shortness_of_Breath', 'Dizziness', 'Chest_Pain', 'Trembling']

for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Encode categorical columns
categorical_columns = ['Gender', 'Trigger', 'Medical_History']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df = df.dropna()

# Features and target
X = df.drop(columns=['Panic_Score'])
y = df['Panic_Score']

# Convert Panic Score into categories
y = pd.cut(y, bins=[0, 3, 6, 10], labels=[0, 1, 2])

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))


# ------------------------------------------------------
#              PREDICTION FUNCTION (USER INPUT)
# ------------------------------------------------------
def predict_panic_score():
    print("\nEnter the following details:")
    user_input = []

    user_input.append(int(input("Age: ")))
    user_input.append(label_encoders['Gender'].transform([input("Gender (Male/Female/Other): ")])[0])
    user_input.append(int(input("Panic attack frequency per month: ")))
    user_input.append(int(input("Duration (minutes): ")))
    user_input.append(label_encoders['Trigger'].transform([input(
        "Trigger (Caffeine, Stress, PTSD, Social Anxiety, Unknown): "
    )])[0])
    user_input.append(int(input("Heart rate (BPM): ")))
    user_input.append(int(input("Sweating (0-No, 1-Yes): ")))
    user_input.append(int(input("Shortness of breath (0-No, 1-Yes): ")))
    user_input.append(int(input("Dizziness (0-No, 1-Yes): ")))
    user_input.append(int(input("Chest pain (0-No, 1-Yes): ")))
    user_input.append(int(input("Trembling (0-No, 1-Yes): ")))

    print("\nValid Medical History options:", list(label_encoders['Medical_History'].classes_))
    medical_input = input("Enter Medical History: ")

    if medical_input in label_encoders['Medical_History'].classes_:
        user_input.append(label_encoders['Medical_History'].transform([medical_input])[0])
    else:
        print("Invalid medical history! Please try again.")
        return

    user_input.append(int(input("Medication (0-No, 1-Yes): ")))
    user_input.append(int(input("Caffeine intake (cups/day): ")))
    user_input.append(int(input("Exercise frequency (days/week): ")))
    user_input.append(float(input("Sleep hours per day: ")))
    user_input.append(int(input("Alcohol consumption (drinks/week): ")))
    user_input.append(int(input("Smoking (0-No, 1-Yes): ")))
    user_input.append(int(input("Therapy (0-No, 1-Yes): ")))

    user_df = pd.DataFrame([user_input], columns=X.columns)
    panic_level = clf.predict(user_df)[0]
    panic_labels = {0: 'Low', 1: 'Medium', 2: 'High'}

    print(f"\nPredicted Panic Attack Severity: {panic_labels[panic_level]}")


decision = input("\nDo you want to enter details to predict panic score? (yes/no): ")
if decision.lower() == 'yes':
    predict_panic_score()
