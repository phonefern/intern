import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(r"D:\project_kiosk\visit_column_ok_filter_depart_encode_r_temp_bi_age_log.csv",encoding="utf-8")
df.dropna(inplace=True)



# df.info()


# X = df[['patient_age','fix_gender_id', 'fever', 'snot', 'sore_throat', 'phlegm', 'stomach_ache', 'Diarrhea', 'vomit', 'headache']]
X = df[['patient_age','temperature_value', 'fix_gender_id', 'fever', 'snot', 'sore_throat', 'phlegm', 'stomach_ache', 'Diarrhea', 'vomit', 'headache'
        , 'itching', 'wound',  'rash', 'edema', 'ear_pain',
          'covid_positive', 'fall', 'leg_pain', 'arm_pain','tumor','bite','tooth']].copy()

y = df['target_department']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

y_pred_dt = model_dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("Decision_Tree_Accurancy:", accuracy_dt)

unique_classes = np.unique(y_test)
# print(unique_classes)



print("---------------------------------------------------------------------------")

ovr_model = OneVsRestClassifier(RandomForestClassifier(random_state=42))

ovr_model.fit(X_train, y_train)

y_scores_ovr = ovr_model.predict_proba(X_test)


# print(y_scores_ovr[:,1])

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# num_classes = len(unique_classes)
# for i in range(num_classes):
#     # print(i)
#     # print(unique_classes[i])
#     # print(y_scores_ovr[:, i])
#     y_scores_ovr[:,i]=replaced_arr = np.where(np.isnan(y_scores_ovr[:,i]), 0, y_scores_ovr[:,i])    
#     fpr[i], tpr[i], _ = roc_curve(y_test == unique_classes[i], y_scores_ovr[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # พล็อต ROC curve สำหรับแต่ละคลาส
# plt.figure(figsize=(8, 6))
# for i in range(num_classes):
#     plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(unique_classes[i], roc_auc[i]))

# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)')
# plt.legend(loc="lower right")
# plt.show()

print("----------------------------------------------------------------------------")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model_rf = RandomForestClassifier(n_estimators=10,random_state=42)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test,y_pred_rf)

print("Random Forest Accurancy :",accuracy_rf)





print("----------------------------------------------------------------------------")

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier()
model_knn.fit(X_train,y_train)

y_pred_knn = model_knn.predict(X_test)

accurancy_knn = accuracy_score(y_test,y_pred_knn)

print("KNN Accurancy: ",accurancy_knn)

print("----------------------------------------------------------------------------")

# from sklearn.svm import SVC

# model_svc = SVC(kernel='linear')
# model_svc.fit(X_train,y_train)

# y_pred_svc = model_svc.predict(X_test)


# accuracy_svc = accuracy_score(y_test,y_pred_svc)

# print("SVC Accurancy:", accuracy_svc)

# print("----------------------------------------------------------------------------")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


model_li = LinearRegression()


model_li.fit(X_train, y_train)


y_pred_li = model_li.predict(X_test)


mse = mean_squared_error(y_test, y_pred_li)
r2 = r2_score(y_test, y_pred_li)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_li, color='blue')
plt.title('Actual vs Predicted (Linear Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print("----------------------------------------------------------------------------")

from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

y_pred_nb = model_nb.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", accuracy_nb)


print("----------------------------------------------------------------------------")


from sklearn.ensemble import GradientBoostingClassifier

model_gbm = GradientBoostingClassifier(random_state=42)
model_gbm.fit(X_train, y_train)

y_pred_bgm = model_gbm.predict(X_test)

accuracy_bgm = accuracy_score(y_test, y_pred_bgm)
print("GBM Accuracy:", accuracy_bgm)
