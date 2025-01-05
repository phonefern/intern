import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np
import time
import os

print('\n---- Reading files ----\n')

# Read data
df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

count_before = df.shape[0]
df = df.dropna()

class_names = ['Emergency & Accident Unit', 'Heart Clinic', 'Neuro Med Center', 'OPD:EYE', 'Dental', 'OPD:MED',
               'OPD:ENT', 'OPD:OBG', 'OPD:Surgery + Uro.', 'Orthopedic Surgery', 'GI Clinic', 'Breast Clinic',
               'Skin & Dermatology']

# Feature selection
X = df[['fix_gender_id', 'patient_age', 'pregnancy', 'fever', 'cough', 'phlegm', 'Diarrhea', 'vomit', 'itching',
        'wound', 'edema', 'tired', 'tumor', 'blood', 'covid_positive', 'headache', 'eye', 'ear_pain', 'snot',
        'tooth', 'sore_throat', 'breast', 'heart', 'rash', 'body', 'stomach_ache', 'vagina', 'urine', 'leg_pain',
        'fall', 'crash', 'bite']].copy()
y = df['target_department']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=24)

lenghtXtrain = len(X_train)
print('Length X_train', lenghtXtrain)
lengthXtest = len(X_test)
print('Length X_test', lengthXtest)

print('\n---- Transforming Sequence Data ----\n')

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print data shapes before SMOTE
print(f"Shape of X_train before SMOTE: {X_train_scaled.shape}")
print(f"Shape of y_train before SMOTE: {y_train.shape}")

# Oversample the minority classes using SMote
smote = SMOTE(random_state=24)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Print data shapes after SMote
print(f"Shape of X_train_resampled after SMOTE: {X_train_resampled.shape}")
print(f"Shape of y_train_resampled after SMOTE: {y_train_resampled.shape}")

print('\n', "----------Defining Model----------", '\n')

# Define the parameter grid for grid search
param_grid = { 
    'penalty': ['l1', 'l2']  # Regularization type
}

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3,  # 3-fold cross-validation
                           verbose=3,
                           n_jobs=-1)  # Use all available CPUs

# Perform grid search on the resampled training data
start_time_gridsearch = time.time()
grid_search.fit(X_train_resampled, y_train_resampled)
end_time_gridsearch = time.time()

# Print grid search results
print("\n---- Grid Search Results ----\n")
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score (Accuracy):", grid_search.best_score_)

# Get the cross-validation results and print the best scores for each fold
cv_results = grid_search.cv_results_
n_splits = grid_search.cv  # Number of splits
for i in range(n_splits):
    best_score = cv_results[f"split{i}_test_score"][grid_search.best_index_]
    print(f"Fold {i+1}: Best Score (Accuracy): {best_score}")

# Print detailed results for each parameter setting
print("\nGrid Search Detailed Results:\n")
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df[['params', 'mean_test_score', 'std_test_score']]
results_df = results_df.sort_values(by='mean_test_score', ascending=False)
print(results_df)

# Get the best model and its hyperparameters
best_log_reg_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predictions and evaluation using the best model found by grid search
start_time_evaluation = time.time()
y_pred_best = best_log_reg_model.predict(X_test_scaled)
y_pred_best_proba = best_log_reg_model.predict_proba(X_test_scaled)
end_time_evaluation = time.time()

accuracy_best = accuracy_score(y_test, y_pred_best)
classification_rep_best = classification_report(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_best_proba, multi_class='ovr')
log_loss_value_best = log_loss(y_test, y_pred_best_proba)
training_time_log_reg = end_time_gridsearch - start_time_gridsearch
evaluation_time_log_reg = end_time_evaluation - start_time_evaluation

print("\n---- Model Evaluation ----\n")
print("Best Parameters:", best_params)
print("Accuracy (Logistic Regression - Best):", accuracy_best)
print("Classification Report (Logistic Regression - Best):\n", classification_rep_best)
print("ROC AUC (Logistic Regression - Best):", roc_auc_best)
print("Log Loss (Logistic Regression - Best):", log_loss_value_best)
print("Training Time (Logistic Regression - Best):", training_time_log_reg, "s.")
print("Evaluation Time (Logistic Regression - Best):", evaluation_time_log_reg, "s.")

# Save the best model if desired
# model_path_best = r"C:\app_build_version_test\my_electron_app_develop\pymodel_log_reg_best.joblib"
# joblib.dump(best_log_reg_model, model_path_best)

# Save results to a text file
model_name = "Logistic_regression"
output_folder = r"./model_save_pic_contribution/save_result_model"
os.makedirs(output_folder, exist_ok=True)
results_path_best = os.path.join(output_folder, f"model_results_Model--{model_name}--Accurancy-{accuracy_best}_best.txt")
with open(results_path_best, "w") as f:
    f.write("Model Evaluation Results\n")
    f.write("========================\n\n")
    f.write(f"Best Parameters: {best_params}\n\n")
    f.write(f"Accuracy (Logistic Regression - Best): {accuracy_best}\n\n")
    f.write("Classification Report (Logistic Regression - Best):\n")
    f.write(f"{classification_rep_best}\n\n")
    f.write(f"ROC AUC (Logistic Regression - Best): {roc_auc_best}\n\n")
    f.write(f"Log Loss (Logistic Regression - Best): {log_loss_value_best}\n\n")
    f.write(f"Training Time (Logistic Regression - Best): {training_time_log_reg} s.\n")
    f.write(f"Evaluation Time (Logistic Regression - Best): {evaluation_time_log_reg} s.\n")
