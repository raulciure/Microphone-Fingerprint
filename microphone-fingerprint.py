# basic_multiclass.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1) Load data (semicolon-separated; no header)
file_path = "Vehicle A - Replay 0x181, 0x161, 0x1a5, Fuzzing 0x244, 0x284, 0x354 - IDs_Datafield_Classification.csv"  # <-- change this to your file
df = pd.read_csv(file_path, sep=';', header=None)

# 2) Split features/labels (last column is class)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3) Train/validation split
X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)

# 4) Build a simple logistic regression
clf = LogisticRegression(max_iter=1000, n_jobs=None)

# 5) Train
clf.fit(X_train, y_train)

# 6) Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))