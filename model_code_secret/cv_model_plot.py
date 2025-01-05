# Import required libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action = "ignore")

# Load the dataset
df = pd.read_csv(r"./updated_visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

# Define the models
models = {
    "XGBoost": xgb.XGBClassifier(random_state=12345, use_label_encoder=False, eval_metric='logloss'),
    "Decision Tree": DecisionTreeClassifier(random_state=12345),
    "Random Forest": RandomForestClassifier(random_state=12345),
    "Logistic Regression": LogisticRegression(random_state=12345),
}

# Define features and target
X = df[['fix_gender_id','patient_age','pregnancy','fever','cough' ,'phlegm', 'Diarrhea', 'vomit', 'itching', 'wound', 'edema', 'tired', 'tumor', 'blood','covid_positive',
            'headache','eye','ear_pain', 'snot', 'tooth','sore_throat','breast','heart','rash','body','stomach_ache','vagina','urine','leg_pain',
            'fall', 'crash','bite']].copy()

y = df['target_department']

# Evaluate each model
results = []
names = []

for name, model in models.items():
    kfold = KFold(n_splits=5, random_state=12345, shuffle=True)  # Added shuffle
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
