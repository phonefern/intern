import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
import joblib
import time
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import numpy as np

lines = '\n-----------------------------------------------------------------------\n'
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

# Prepare data for XGBoost
dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# XGBoost parameters
params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y_train)),
    'learning_rate': 0.1,
    'tree_method': 'hist',
    'device': 'cuda',
    'n_estimators' : 100
}

# Grid Search for hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    
}

cv = GridSearchCV(estimator=xgb.XGBClassifier(**params), param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
cv.fit(X_train_resampled, y_train_resampled)

print("Best parameters found: ", cv.best_params_)
print("Best accuracy found: ", cv.best_score_)

# Save best parameters to a text file
best_params_path = './model_save_pic_contribution/save_result_model/best_parameters.txt'
with open(best_params_path, "w") as f:
    f.write("Best Parameters Found\n")
    f.write("======================\n")
    f.write(f"Best parameters found: {cv.best_params_}\n")
    f.write(f"Best accuracy found: {cv.best_score_}\n")

# Update params with the best found parameters
params.update(cv.best_params_)

num_round = 10  # Total number of training rounds
early_stopping_rounds = 50  # Number of rounds to trigger early stopping
models = []  # List to save models

# Training with loop
start_time = time.time()
for i in range(num_round):
    print(f"Training round {i+1}/{num_round}")
    model = xgb.train(params, dtrain, num_boost_round=1, xgb_model=None if i == 0 else models[-1])
    models.append(model)
    
    # Predictions and evaluation
    y_pred_gb_proba = model.predict(dtest)
    y_pred_gb = np.argmax(y_pred_gb_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_gb)
    roc_auc = roc_auc_score(y_test, y_pred_gb_proba, multi_class='ovr')
    print(f"Accuracy after round {i+1}: {accuracy}")
    print(f"ROC AUC after round {i+1}: {roc_auc}")
    
    # Early stopping
    if i > early_stopping_rounds and all(accuracy <= accuracy_score(y_test, np.argmax(models[-j-1].predict(dtest), axis=1)) for j in range(1, early_stopping_rounds+1)):
        print("Early stopping")
        break

end_time = time.time()
training_time_gb = end_time - start_time

# Final evaluation
y_pred_gb_proba = models[-1].predict(dtest)
y_pred_gb = np.argmax(y_pred_gb_proba, axis=1)
accuracy = accuracy_score(y_test, y_pred_gb)
classification_rep = classification_report(y_test, y_pred_gb)
roc_auc = roc_auc_score(y_test, y_pred_gb_proba, multi_class='ovr')
log_loss_value = log_loss(y_test, y_pred_gb_proba)

print("Accuracy (Gradient Boosting with GPU):", accuracy)
print("Classification Report (Gradient Boosting with GPU):\n", classification_rep)
print("ROC AUC (Gradient Boosting with GPU):", roc_auc)
print("Log Loss (Gradient Boosting with GPU):", log_loss_value)
print("Training Time (Gradient Boosting with GPU):", training_time_gb, "s.")

# Save the final model
# model_path = r"C:\app_build_version_test\my_electron_app_develop\pymodel_gb_xgb.joblib"
# joblib.dump(models[-1], model_path)

# Save results to a text file
results_path = f'./model_save_pic_contribution/save_result_model\model_results_Accurancy-{accuracy}.txt'
with open(results_path, "w") as f:
    f.write("Model Evaluation Results\n")
    f.write("========================\n\n")
    f.write(f"จำนวนข้อมูลที่ถูกลบออก: {count_dropped}\n\n")
    f.write(f"Accuracy (Gradient Boosting with GPU): {accuracy}\n\n")
    f.write(f"ROC AUC (Gradient Boosting with GPU): {roc_auc}\n\n")
    f.write("Classification Report (Gradient Boosting with GPU):\n")
    f.write(f"{classification_rep}\n\n")
    f.write(f"Log Loss (Gradient Boosting with GPU): {log_loss_value}\n\n")
    f.write(f"Training Time (Gradient Boosting with GPU): {training_time_gb} s.\n")
