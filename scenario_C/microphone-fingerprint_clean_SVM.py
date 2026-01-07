import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1) CONFIGURATION
# Point this to the folder containing the 3 sound type folders
# Structure expected:
#   dataset/
#       Horn/
#           iPhone13/ (files...)
#           Pixel6/   (files...)
#       Wipers/
#           iPhone13/ (files...)
#           ...

# dataset_root_path = "../data/clean_recordings/live_recordings"
dataset_root_path = os.path.join(os.path.dirname(__file__), os.pardir, "data/clean_recordings/live_recordings")

data_rows = []
labels = []

print("Scanning folder structure...")

# 2) CRAWL FOLDERS RECURSIVELY
# os.walk will go: Root -> Horn -> iPhone13 -> (Files)
for root, dirs, files in os.walk(dataset_root_path):
    # Skip folders that have no files (like the "Horn" container folder itself)
    if not files:
        continue
    
    # The label is the name of the immediate folder containing the files
    # e.g., if path is ".../Horn/iPhone13", basename is "iPhone13"
    current_phone_label = os.path.basename(root)
    
    # Optional: Print what we are currently processing to track progress
    # We can also capture the sound type from the parent folder if needed for debugging
    parent_folder = os.path.basename(os.path.dirname(root))
    print(f"Processing: {parent_folder} -> {current_phone_label} ({len(files)} files)")
    
    for filename in files:
        file_path = os.path.join(root, filename)
        
        # # simple check to ensure we only read data files
        # if not (filename.endswith('.csv') or filename.endswith('.txt')):
        #     continue

        try:
            # 3) LOAD AND TRANSPOSE
            # Read the file (assuming no header, 4096 lines)
            single_file_df = pd.read_csv(file_path, sep=',', header=None)
            
            # Validation check
            # if single_file_df.shape[0] != 4096:
            #     continue

            # Flatten 4096 vertical lines into 1 horizontal row
            flat_data = single_file_df.values.flatten()
            
            data_rows.append(flat_data)
            labels.append(current_phone_label)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# 4) CREATE MASTER DATASET
print("\nCompiling dataset...")
X = pd.DataFrame(data_rows)
y = pd.Series(labels)

print(f"Final Dataset Shape: {X.shape} (Rows/Examples, Columns/Frequencies)")
print(f"Classes found: {y.unique()}")

# 5) SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6) PREPROCESSING
# Standard Scaling is CRUCIAL for SVM to work correctly
scaler = StandardScaler()
print("Scaling data...")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7) TRAIN SVM MODEL
# Using Linear Kernel as recommended for high-dimensional (4096 features) data
print("\nTraining Support Vector Machine (this may take a minute)...")
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train_scaled, y_train)

# 8) EVALUATE
print("\nEvaluating...")
y_pred = clf.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 9) CONFUSION MATRIX
# This is helpful to see if the model confuses specific phones
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))