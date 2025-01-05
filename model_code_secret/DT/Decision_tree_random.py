import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import time
from scipy.stats import randint

def run_model_with_smote(use_smote=True):
    print('\n---- Reading files ----\n')

    # Read data
    df = pd.read_csv(r"./updated_visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

    count_before = df.shape[0]
    df = df.dropna()

    class_names = ['Emergency & Accident Unit', 'Heart Clinic', 'Neuro Med Center', 'OPD:EYE', 'Dental', 'OPD:MED',
                   'OPD:ENT', 'OPD:OBG', 'OPD:Surgery + Uro.', 'Orthopedic Surgery', 'GI Clinic', 'Breast Clinic',
                   'Skin & Dermatology']

    # Feature selection
    X = df.drop("target_department", axis=1)  # Independent variables
    y = df.target_department  # Dependent variable

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=24)

    print('Length X_train:', len(X_train))
    print('Length X_test:', len(X_test))

    print('\n---- Transforming Sequence Data ----\n')

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print data shapes before SMOTE
    print(f"Shape of X_train before SMOTE: {X_train_scaled.shape}")
    print(f"Shape of y_train before SMOTE: {y_train.shape}")

    if use_smote:
        # Oversample the minority classes using SMOTE
        smote = SMOTE(random_state=24)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Print data shapes after SMOTE
        print(f"Shape of X_train_resampled after SMOTE: {X_train_resampled.shape}")
        print(f"Shape of y_train_resampled after SMOTE: {y_train_resampled.shape}")
    else:
        X_train_resampled, y_train_resampled = X_train_scaled, y_train

    print('\n', "----------Defining Model----------", '\n')

    # Define the parameter distributions for Randomized Search
    param_distributions = {
        'criterion': ['entropy'],
        'max_depth': [19],  # Randomly selects integer values between 1 and 20
        'min_samples_split': [3],  # Randomly selects integer values between 2 and 20
        'min_samples_leaf': [2]  # Randomly selects integer values between 1 and 10
    }

    # Perform Randomized Search
    random_cv = RandomizedSearchCV(estimator=DecisionTreeClassifier(), 
                                   param_distributions=param_distributions, 
                                   n_iter=50,  # Number of parameter settings that are sampled
                                   scoring='accuracy', 
                                   cv=5,  # 3-fold cross-validation
                                   verbose=3, 
                                   random_state=24, 
                                   n_jobs=-1)  # Use all available CPUs

    # Fit the Randomized Search on the resampled training data
    start_time_randomsearch = time.time()
    random_cv.fit(X_train_resampled, y_train_resampled)
    end_time_randomsearch = time.time()

    # Print Randomized Search results
    print("\n---- Randomized Search Results ----\n")
    print("Best Parameters:", random_cv.best_params_)
    print("Best Cross-validation Score (Accuracy):", random_cv.best_score_)

    # Get the best model and its hyperparameters
    best_decision_tree_model = random_cv.best_estimator_
    best_params = random_cv.best_params_

    # Print best scores for each fold
    cv_results = random_cv.cv_results_
    for i in range(random_cv.cv):
        fold_scores = cv_results[f'split{i}_test_score']
        best_fold_score = np.max(fold_scores)
        print(f"Fold {i+1}: Best Score: {best_fold_score:.4f}")

    # Predictions and evaluation using the best model found by Randomized Search
    start_time_evaluation = time.time()
    y_pred_best = best_decision_tree_model.predict(X_test_scaled)
    y_pred_best_proba = best_decision_tree_model.predict_proba(X_test_scaled)
    end_time_evaluation = time.time()

    accuracy_best = accuracy_score(y_test, y_pred_best)
    classification_rep_best = classification_report(y_test, y_pred_best, target_names=class_names)
    roc_auc_best = roc_auc_score(y_test, y_pred_best_proba, multi_class='ovr')
    log_loss_value_best = log_loss(y_test, y_pred_best_proba)
    training_time_decision_tree = end_time_randomsearch - start_time_randomsearch
    evaluation_time_decision_tree = end_time_evaluation - start_time_evaluation

    print("\n---- Model Evaluation ----\n")
    print("Best Parameters:", best_params)
    print("Accuracy (Decision Tree - Best):", accuracy_best)
    print("Classification Report (Decision Tree - Best):\n", classification_rep_best)
    print("ROC AUC (Decision Tree - Best):", roc_auc_best)
    print("Log Loss (Decision Tree - Best):", log_loss_value_best)
    print("Training Time (Decision Tree - Best):", training_time_decision_tree, "s.")
    print("Evaluation Time (Decision Tree - Best):", evaluation_time_decision_tree, "s.")

    # Save results to a text file
    model_name = "Decision_tree_random_search" + ("_with_SMOTE" if use_smote else "_without_SMOTE")
    results_path_best = rf'D:\project_kiosk\model_save_pic_contribution\save_result_model\model_results_Model--{model_name}--Accuracy-{accuracy_best}_best.txt'
    with open(results_path_best, "w") as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n\n")
        f.write(f"Best Parameters: {best_params}\n\n")
        f.write(f"Accuracy (Decision Tree - Best): {accuracy_best}\n\n")
        f.write("Classification Report (Decision Tree - Best):\n")
        f.write(f"{classification_rep_best}\n\n")
        f.write(f"ROC AUC (Decision Tree - Best): {roc_auc_best}\n\n")
        f.write(f"Log Loss (Decision Tree - Best): {log_loss_value_best}\n\n")
        f.write(f"Training Time (Decision Tree - Best): {training_time_decision_tree} s.\n")
        f.write(f"Evaluation Time (Decision Tree - Best): {evaluation_time_decision_tree} s.\n")

# Run model with SMOTE
# run_model_with_smote(use_smote=True)

# Run model without SMOTE
run_model_with_smote(use_smote=False)
