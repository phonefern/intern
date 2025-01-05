import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# โหลดข้อมูล
df = pd.read_csv(r"./updated_visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")
count_before = df.shape[0]
df = df.dropna()

class_names = ['Emergency & Accident Unit','Heart Clinic','Neuro Med Center','OPD:EYE','Dental','OPD:MED','OPD:ENT','OPD:OBG','OPD:Surgery + Uro.','Orthopedic Surgery','GI Clinic','Breast Clinic',
                    'Skin & Dermatology']

# ฟีเจอร์และเป้าหมาย
X = df[['fix_gender_id', 'patient_age', 'pregnancy', 'fever', 'cough', 'phlegm', 'Diarrhea', 'vomit', 'itching', 
        'wound', 'edema', 'tired', 'tumor', 'blood', 'covid_positive', 'headache', 'eye', 'ear_pain', 
        'snot', 'tooth', 'sore_throat', 'breast', 'heart', 'rash', 'body', 'stomach_ache', 'vagina', 
        'urine', 'leg_pain', 'fall', 'crash', 'bite']].copy()
y = df['target_department']

# แบ่งข้อมูลเป็นชุดฝึกและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# มาตรฐานข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้างโมเดล Logistic Regression พร้อม L1 penalty
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
lasso_model.fit(X_train_scaled, y_train)

# ทำนายผลและประเมินผล
y_pred = lasso_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=class_names))

# ดูค่าสัมประสิทธิ์
print("Coefficients:", lasso_model.coef_)
print("Intercept:", lasso_model.intercept_)
