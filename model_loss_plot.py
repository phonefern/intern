import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve
import joblib
import time
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read data
df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

count_before = df.shape[0]
df = df.dropna()
count_after = df.shape[0]
count_dropped = count_before - count_after
print("จำนวนข้อมูลที่ถูกลบออก:", count_dropped)

# Feature selection
X = df[['fix_gender_id','patient_age','pregnancy','fever','cough' ,'phlegm', 'Diarrhea', 'vomit', 'itching', 'wound', 'edema', 'tired', 'tumor', 'blood','covid_positive',
        'headache','eye','ear_pain', 'snot', 'tooth','sore_throat','breast','heart','rash','body','stomach_ache','vagina','urine','leg_pain',
        'fall', 'crash','bite']].copy()
y = df['target_department']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=24)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversample the minority classes using SMOTE
smote = SMOTE(random_state=24)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Convert data to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Define parameters for XGBoost
params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_encoded)),
    'learning_rate': 0.2,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'tree_method': 'gpu_hist',
    'eval_metric': 'mlogloss'
}

# Train the model with GPU
start_time = time.time()
evals_result = {}
gb_model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dtest, 'eval')], evals_result=evals_result, verbose_eval=False)
end_time = time.time()

# Predictions and evaluation
y_pred_gb_proba = gb_model.predict(dtest)

# Select class with highest probability
y_pred_gb = np.argmax(y_pred_gb_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred_gb)
classification_rep = classification_report(y_test, y_pred_gb, target_names=class_names)
roc_auc = roc_auc_score(y_test, y_pred_gb_proba, multi_class='ovo')
log_loss_value = log_loss(y_test, y_pred_gb_proba)
training_time_gb = end_time - start_time

print("Accuracy (Gradient Boosting with GPU):", accuracy)
print("Classification Report (Gradient Boosting with GPU):\n", classification_rep)
print("ROC AUC (Gradient Boosting with GPU):", roc_auc)
print("Log Loss (Gradient Boosting with GPU):", log_loss_value)
print("Training Time (Gradient Boosting with GPU):", training_time_gb, "s.")

# Plot Logarithmic Loss (Log Loss or Cross-Entropy Loss)
epochs = len(evals_result['train']['mlogloss'])
x_axis = range(0, epochs)
plt.figure(figsize=(12, 8))
plt.plot(x_axis, evals_result['train']['mlogloss'], label='Train')
plt.plot(x_axis, evals_result['eval']['mlogloss'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss Over Epochs')
plt.legend()
plt.show()

# Plot Confusion Matrix with threshold 0.4
threshold = 0.4
y_pred_gb_thresh = np.argmax(y_pred_gb_proba, axis=1) if threshold == 0.4 else y_pred_gb

cm = confusion_matrix(y_test, y_pred_gb_thresh)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Threshold 0.4')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Save the model
# model_path = r"C:\app_build_version_test\my_electron_app_develop\pymodel_gb_xgb.joblib"
# joblib.dump(gb_model, model_path)

# Save results to a text file
results_path = f'./model_save_pic_contribution/save_result_model\model_results_Accurancy-{accuracy}.txt"
with open(results_path, "w") as f:
    f.write("Model Evaluation Results\n")
    f.write("========================\n\n")
    f.write(f"จำนวนข้อมูลที่ถูกลบออก: {count_dropped}\n\n")
    f.write(f"Accuracy (Gradient Boosting with GPU): {accuracy}\n\n")
    f.write("Classification Report (Gradient Boosting with GPU):\n")
    f.write(f"{classification_rep}\n\n")
    f.write(f"ROC AUC (Gradient Boosting with GPU): {roc_auc}\n\n")
    f.write(f"Log Loss (Gradient Boosting with GPU): {log_loss_value}\n\n")
    f.write(f"Training Time (Gradient Boosting with GPU): {training_time_gb} s.\n")
