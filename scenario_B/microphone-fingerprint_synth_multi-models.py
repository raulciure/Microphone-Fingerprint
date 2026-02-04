import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, log_loss

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# For nice graphical plots
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


# dataset_root_path = ".data/clean_recordings/synthetically_reproduced_environmental_sound"
dataset_root_path = os.path.join(os.path.dirname(__file__), os.pardir, "data/clean_recordings/synthetically_reproduced_environmental_sound")

data_rows = []
labels = []

print("Scanning folder structure...")

# 2) CRAWL FOLDERS RECURSIVELY
for root, dirs, files in os.walk(dataset_root_path):
    # Skip folders that don't have files
    if not files:
        continue
    
    # The label is the name of the immediate folder containing the files (e.g. S6 - scenario C or A - scenario B)
    current_mic_label = os.path.basename(root)
    
    # Print data loading and processing progress
    parent_folder = os.path.basename(os.path.dirname(root))
    print(f"Processing: {parent_folder} -> {current_mic_label} ({len(files)} files)")
    
    for filename in files:
        file_path = os.path.join(root, filename)

        try:
            # Read the file (assuming no header, 4096 lines)
            single_file_df = pd.read_csv(file_path, sep=',', header=None)

                # Flatten 4096 vertical lines into 1 horizontal row
                # flat_data = single_file_df.values.flatten()

            # IMPORTANT CHANGE:
            # We select all rows (:), but ONLY the second column (1)
            # The first column (0) is Frequency, which we discard.
            amplitudes = single_file_df.iloc[:, 1].values
            
            # Add amplitudes and label to the lists
            data_rows.append(amplitudes)
            labels.append(current_mic_label)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Create main dataset
print("\nCompiling dataset...")
X = pd.DataFrame(data_rows)
y = pd.Series(labels)

print(f"Final Dataset Shape: {X.shape} (Measurments, Frequencies)")
print(f"Classes found: {y.unique()} | Number of distinct classes: {len(y.unique())}")

# 5) SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6) PREPROCESSING
# Standard Scaling is CRUCIAL for SVM to work correctly
scaler = StandardScaler()
print("Scaling data...")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3) DEFINE MODELS ---
# We store them in a dictionary to loop through them easily
models = {
    "SVM (Linear)": SVC(kernel='linear', C=1.0, probability=True),
    
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    
    "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# --- 4) THE TOURNAMENT LOOP ---
print("\n--- STARTING MODEL COMPARISON ---\n")

results = []

# Train each model
model: SVC | RandomForestClassifier | MLPClassifier | KNeighborsClassifier
for name, model in models.items():
    print(f"Training {name}...")

    # KNN and Neural Nets strictly need scaled data. 
    # Random Forest technically doesn't, but it doesn't hurt.
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)

    # Probability prediction
    y_probs = model.predict_proba(X_test_scaled)
    
    # Score
    acc = accuracy_score(y_test, y_pred)
    clasif_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
        # roc = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
        # prec_rec_curve = precision_recall_curve(y_test, y_probs)
    lg_ls = log_loss(y_test, y_probs)

    # Get the list of class names (e.g., ['iPhone13', 'Pixel6', ...])
    class_labels = model.classes_ 
    # Create a DataFrame
    cm_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
        # roc_df = pd.DataFrame(roc, index=class_labels, columns=class_labels)
        # prec_rec_curve_df = pd.DataFrame(prec_rec_curve, index=class_labels, columns=class_labels)
    
    print(f"--> {name} Accuracy: {acc:.4f}\n")
    print(f"--> {name} Classification Report:\n{clasif_report}")
    print(f"--> {name} Confusion Matrix:")
    print(cm_df + "\n")

                # # Confusion Matrix
                #     # skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, title="Confusion Matrix")
                #     # plt.show()
                # conf_matrix = yelbrk.classifier.ConfusionMatrix(model, classes=model.classes_)
                # conf_matrix.score(X_test_scaled, y_test)
                # conf_matrix.show()

                # print(f"--> {name} ROC Curves:")
                # # ROC Curves
                #     # skplt.metrics.plot_roc(y_test, y_probs, title="ROC Curves per phone")
                #     # plt.show()
                # roc_curve = yelbrk.classifier.ROCAUC(model, classes=model.classes_)
                # roc_curve.score(X_test_scaled, y_test)
                # roc_curve.show()

                # print(f"--> {name} Precision-Recall Curve:")
                # # Precision-Recall curve
                #     # skplt.metrics.plot_precision_recall(y_test, y_probs, title="Precision-Recall Curve")
                #     # plt.show()
                # pr_curve = yelbrk.classifier.PrecisionRecallCurve(model, classes=model.classes_)
                # pr_curve.score(X_test_scaled, y_test)
                # pr_curve.show()

    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test_scaled,
        y_test,
        cmap=plt.colormaps.get_cmap("Blues"),
        normalize='true'
    )
    disp.ax_.set_title(f"{name} - confusion matrix:")
    plt.show()

                    # disp = RocCurveDisplay.from_estimator(
                    #     model,
                    #     X_test_scaled,
                    #     y_test
                    # )
                    # disp.ax_.set_title(f"{name} - ROC curve:")
                    # plt.show()

                    # disp = PrecisionRecallDisplay.from_estimator(
                    #     model,
                    #     X_test_scaled,
                    #     y_test
                    # )
                    # disp.ax_.set_title(f"{name} - Precision-Recall curve:")
                    # plt.show()

        # print(f"--> {name} ROC Curve:")
        # print(roc_df)

    print(f"--> {name} ROC AUC Score: {roc_auc:.4f}\n")

        # print(f"--> {name} Precission-Recall Curve:")
        # print(prec_rec_curve_df)

    print(f"--> {name} Log Loss: {lg_ls:.4f}\n")

    print()
    results.append({'Model': name, 'Accuracy': acc})

# --- 5) SUMMARY TABLE ---
print("\n--- FINAL RESULTS ---")
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print(results_df)