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
import os
import seaborn as sns

# Read data
df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

count_before = df.shape[0]

df = df.dropna()

count_after = df.shape[0]

count_dropped = count_before - count_after

print("จำนวนข้อมูลที่ถูกลบออก:", count_dropped)

class_names = ['Emergency & Accident Unit','Heart Clinic','Neuro Med Center','OPD:EYE','Dental','OPD:MED','OPD:ENT','OPD:OBG','OPD:Surgery + Uro.','Orthopedic Surgery','GI Clinic','Breast Clinic',
                'Skin & Dermatology']

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

# Train the model with GPU
start_time = time.time()
gb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.2, objective='multi:softprob', use_label_encoder=False, tree_method='gpu_hist', colsample_bytree=1, max_depth=7, min_child_weight=1, subsample=0.9)
gb_model.fit(X_train_resampled, y_train_resampled)
end_time = time.time()

# Predictions and evaluation
y_pred_gb_proba = gb_model.predict_proba(X_test_scaled)

# Select class with highest probability
y_pred_gb = np.argmax(y_pred_gb_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred_gb)
classification_rep = classification_report(y_test, y_pred_gb)
roc_auc = roc_auc_score(y_test, y_pred_gb_proba, multi_class='ovo')
log_loss_value = log_loss(y_test, y_pred_gb_proba)
training_time_gb = end_time - start_time

print("Accuracy (Gradient Boosting with GPU):", accuracy)
print("Classification Report (Gradient Boosting with GPU):\n", classification_rep)
print("ROC AUC (Gradient Boosting with GPU):", roc_auc)
print("Log Loss (Gradient Boosting with GPU):", log_loss_value)
print("Training Time (Gradient Boosting with GPU):", training_time_gb, "s.")


# # Plot Confusion Matrix with threshold 0.4
# threshold = 0.4
# y_pred_gb_thresh = np.argmax(y_pred_gb_proba, axis=1) if threshold == 0.4 else y_pred_gb

# cm = confusion_matrix(y_test, y_pred_gb_thresh)
# plt.figure(figsize=(12, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix with Threshold 0.4')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.show()


# Calibration Curve
# plt.figure(figsize=(10, 10))
# for i in range(y_pred_gb_proba.shape[1]):
#     prob_true, prob_pred = calibration_curve(y_test == i, y_pred_gb_proba[:, i], n_bins=10)
#     plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
    
# plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
# plt.xlabel('Predicted probability')
# plt.ylabel('True probability')
# plt.title('Calibration curves')
# plt.legend()
# plt.show()

# Save the model
model_path = r"D:\project_kiosk\model_code_secret\model_joblib\pymodel_gb_xgb.joblib_v1.0"
joblib.dump(gb_model, model_path)

# Save results to a text file
results_path = f'./model_save_pic_contribution/save_result_model\model_results_Accurancy-{accuracy}.txt'
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



# # Load the model
# gb_model_loaded = joblib.load(joblib_file)

# # Predict with the loaded model
# y_pred_gb_loaded = gb_model_loaded.predict(X_test)
# print("Accuracy (Loaded Gradient Boosting with GPU):", accuracy_score(y_test, y_pred_gb_loaded))
# print("Classification Report (Loaded Gradient Boosting with GPU):\n", classification_report(y_test, y_pred_gb_loaded,target_names=target_names))

########################################################################################################################################################################################

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report
# import time
# import xgboost as xgb
# import json

# # Read data
# df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

# # Feature selection
# X = df[['fix_gender_id','patient_age','temperature_value','fever','cough' ,'phlegm', 'Diarrhea', 'vomit', 'itching', 'wound', 'edema', 'tired', 'tumor', 'blood','covid_positive',
#         'headache','eye','ear_pain', 'snot', 'tooth','sore_throat','breast','heart','rash','body','stomach_ache','vagina','urine','leg_pain',
#         'fall', 'crash','bite']].copy()
# y = df['target_department']

# # Encode labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=24)

# # Train the model with GPU
# start_time = time.time()
# gb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, objective='multi:softprob', use_label_encoder=False, tree_method='hist')
# gb_model.fit(X_train, y_train)
# end_time = time.time()

# # Predictions and evaluation
# y_pred_gb = gb_model.predict(X_test)
# target_names = [str(cls) for cls in label_encoder.classes_]
# print("Accuracy (Gradient Boosting with GPU):", accuracy_score(y_test, y_pred_gb))
# print("Classification Report (Gradient Boosting with GPU):\n", classification_report(y_test, y_pred_gb, target_names=target_names))

# training_time_gb = end_time - start_time
# print("Training Time (Gradient Boosting with GPU):", training_time_gb, "s.")

# # Save the model in JSON format
# json_file = (r"D:\project_kiosk\my_electron_app2\pymodel_gb_xgb.json")
# gb_model.save_model(json_file)

# # Load the model from JSON
# gb_model_loaded = xgb.XGBClassifier()
# gb_model_loaded.load_model(json_file)

# # Predict with the loaded model
# y_pred_gb_loaded = gb_model_loaded.predict(X_test)
# print("Accuracy (Loaded Gradient Boosting with GPU):", accuracy_score(y_test, y_pred_gb_loaded))
# print("Classification Report (Loaded Gradient Boosting with GPU):\n", classification_report(y_test, y_pred_gb_loaded, target_names=target_names))

#################################################################################################################################################################################################


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import time
# from sklearn.tree import DecisionTreeClassifier

# # Read data
# df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

# count_before = df.shape[0]

# df = df.dropna()

# count_after = df.shape[0]

# count_dropped = count_before - count_after

# print("จำนวนข้อมูลที่ถูกลบออก:", count_dropped)

# # Feature selection
# X = df[['fix_gender_id','patient_age','pregnancy','temperature_value','fever', 'phlegm', 'Diarrhea', 'vomit', 'itching', 'wound', 'rash', 'edema', 'tired',
#         'headache','eye','ear_pain', 'snot', 'tooth','sore_throat','breast','heart','body','stomach_ache','vagina','urine','leg_pain','fall', 'tumor', 'crash', 'blood',
#         'covid_positive','bite']].copy()

# y = df['target_department']




# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# # Train the Decision Tree model
# start_time = time.time()
# dt_model = DecisionTreeClassifier(random_state=24)
# dt_model.fit(X_train, y_train)
# end_time = time.time()

# # Predictions and evaluation
# y_pred_dt = dt_model.predict(X_test)
# print("Accuracy (Decision Tree):", accuracy_score(y_test, y_pred_dt))
# print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# training_time_dt = end_time - start_time
# print("Training Time (Decision Tree):", training_time_dt, "s.")

# # Save the model
# joblib_file = r"D:\project_kiosk\web_kiosk\pymodel_dt.joblib"
# joblib.dump(dt_model, joblib_file)
