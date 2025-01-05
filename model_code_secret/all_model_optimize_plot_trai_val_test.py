import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss, brier_score_loss, roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
import time
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import os
import joblib

def train_model(use_smote=True):
    # Read data
    df = pd.read_csv(r"./updated_visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

    count_before = df.shape[0]
    df = df.dropna()
    count_after = df.shape[0]
    count_dropped = count_before - count_after
    print("จำนวนข้อมูลที่ถูกลบออก:", count_dropped)

    # Feature selection
    X = df.drop(columns=['target_department'])  # Drop the target column
    y = df['target_department']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into 70% train, 15% validation, and 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=24)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=24)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE if specified
    if use_smote:
        smote = SMOTE(random_state=24)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    print(f"Shape of X_train: {X_train_resampled.shape}")
    print(f"Shape of X_test: {y_train_resampled.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    # Define models
    models = {
        "XGBoost": xgb.XGBClassifier(n_estimators=500, learning_rate=0.2, objective='multi:softprob', use_label_encoder=False, tree_method='gpu_hist', colsample_bytree=1, max_depth=7, min_child_weight=1, subsample=0.9, gamma=0),
        "Decision Tree": DecisionTreeClassifier(random_state=24, criterion='gini', max_depth=20, min_samples_leaf=10, min_samples_split=2),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=24, max_depth=None, min_samples_split=10, min_samples_leaf=2),
        "Naive Bayes": GaussianNB(var_smoothing=0.001),
        "Logistic Regression": LogisticRegression(multi_class='ovr', solver='liblinear', random_state=24, penalty='l1')
    }

    results = {}
    fpr = {}
    tpr = {}
    roc_auc = {}
    class_names = label_encoder.classes_

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train_resampled, y_train_resampled)
        end_time = time.time()

        y_pred_proba = model.predict_proba(X_test_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)  # Get the predicted class with the highest probability
        
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        roc_auc_score_value = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
        log_loss_value = log_loss(y_test, y_pred_proba)
        training_time = end_time - start_time

        # Calculate Brier Score for multiclass
        brier_scores = []
        for i in range(y_pred_proba.shape[1]):
            brier_scores.append(brier_score_loss((y_test == i).astype(int), y_pred_proba[:, i]))
        brier_score = np.mean(brier_scores)
        
        results[name] = {
            "accuracy": accuracy,
            "classification_report": classification_rep,
            "roc_auc": roc_auc_score_value,
            "log_loss": log_loss_value,
            "training_time": training_time,
            "y_pred_proba": y_pred_proba,
            "brier_score": brier_score
        }
        
        fpr[name] = {}
        tpr[name] = {}
        roc_auc[name] = {}
        sensitivities = []
        specificities = []
        
        for i, class_name in enumerate(class_names):
            fpr[name][i], tpr[name][i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            roc_auc[name][class_name] = auc(fpr[name][i], tpr[name][i])
            
            # Calculate confusion matrix for each class
            conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
            tp = conf_matrix[i, i]
            fn = conf_matrix[i, :].sum() - tp
            fp = conf_matrix[:, i].sum() - tp
            tn = conf_matrix.sum() - (tp + fn + fp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        # Calculate average sensitivity and specificity
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)
        
        print(f"Accuracy ({name}):", accuracy)
        print(f"ROC AUC ({name}):", roc_auc_score_value)
        print(f"Log Loss ({name}):", log_loss_value)
        print(f"Brier Score ({name}):", brier_score)
        print(f"Average Sensitivity ({name}):", avg_sensitivity)
        print(f"Average Specificity ({name}):", avg_specificity)
        print(f"Training Time ({name}):", training_time, "s.")
        print(f"Classification Report ({name}) :", classification_rep)
        
        print("\n")

    plt.figure(figsize=(15, 10))

    for name, result in results.items():
        y_pred_proba = result["y_pred_proba"]
        n_classes = y_pred_proba.shape[1]
        
        # Initialize lists to store the mean predicted probabilities and fraction of positives for each class
        all_prob_true = []
        all_prob_pred = []
        
        for i in range(n_classes):
            # Binarize y_test for the current class
            y_test_binary = (y_test == i).astype(int)
            
            # Get predicted probabilities for the current class
            y_pred_class_proba = y_pred_proba[:, i]
            
            # Calculate calibration curve for the current class
            prob_true, prob_pred = calibration_curve(y_test_binary, y_pred_class_proba, n_bins=10)
            
            # Ensure the lengths of prob_true and prob_pred are the same for each class
            all_prob_true.append(np.interp(np.linspace(0, 1, num=10), prob_pred, prob_true))
            all_prob_pred.append(np.linspace(0, 1, num=10))
        
        # Calculate the mean predicted probabilities and fraction of positives across all classes
        mean_prob_true = np.mean(all_prob_true, axis=0)
        mean_prob_pred = np.mean(all_prob_pred, axis=0)
        
        # Plot the mean calibration curve
        plt.plot(mean_prob_pred, mean_prob_true, marker='o', label=f'{name} - Mean Probability')

    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve using mean probability across all classes')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Compute average ROC AUC across all classes for each model
    average_fpr = {}
    average_tpr = {}
    average_roc_auc = {}

    for name in models.keys():
        all_fpr = np.unique(np.concatenate([fpr[name][i] for i in range(len(class_names))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_names)):
            mean_tpr += np.interp(all_fpr, fpr[name][i], tpr[name][i])
        mean_tpr /= len(class_names)
        
        average_fpr[name] = all_fpr
        average_tpr[name] = mean_tpr
        average_roc_auc[name] = auc(average_fpr[name], average_tpr[name])
        
        # Print metrics for transparency
        print(f"{name}:")
        print(f"  Accuracy: {results[name]['accuracy']:.2f}")
        for class_name, auc_score in roc_auc[name].items():
            print(f"  {class_name}: AUC = {auc_score:.2f}")
        print(f"  Average AUC: {average_roc_auc[name]:.2f}")
        print(f"  Brier Score: {results[name]['brier_score']:.2f}")
        print()

    # Save results to a text file
    output_folder = r"./model_save_pic_contribution"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "model_performance_all_model.txt")

    with open(output_file, "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n\n")
        f.write(f"จำนวนข้อมูลที่ถูกลบออก: {count_dropped}\n\n")
        for name in models.keys():
            f.write(f"{name}:\n")
            f.write(f"  Accuracy: {results[name]['accuracy']:.2f}\n")
            for class_name, auc_score in roc_auc[name].items():
                f.write(f"  {class_name}: AUC = {auc_score:.2f}\n")
            f.write(f"  Average AUC: {average_roc_auc[name]:.2f}\n")
            f.write(f"  Brier Score: {results[name]['brier_score']:.2f}\n")
            f.write(f"  Average Sensitivity: {avg_sensitivity:.2f}\n")
            f.write(f"  Average Specificity: {avg_specificity:.2f}\n")
            f.write(f"  Training Time: {results[name]['training_time']} s.\n")
            f.write("\n")

    # Plot ROC curves for each model
    plt.figure(figsize=(8, 6))

    for name in models.keys():
        plt.plot(average_fpr[name], average_tpr[name], lw=2, label='%s (AUC = %0.2f)' % (name, average_roc_auc[name]))

    # Plot random guessing line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random Guessing')

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - All Models')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# Example of usage:
train_model(use_smote=False)
