import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age.csv",encoding='utf-8')
pd.set_option("display.max_rows",None)
# min_age = df['patient_age'].min()
# max_age = df['patient_age'].max()

# df['patient_age'] = (df['patient_age'] - min_age) / (max_age - min_age)

# print(df['patient_age'].head(1000))

# # กำหนดขนาดกราฟ
# plt.figure(figsize=(10, 6))

# # สร้าง histogram ของข้อมูล
# sns.histplot(data=df, x='patient_age', kde=True, color='skyblue', bins=120)

# # เพิ่มชื่อแกน x, y และชื่อกราฟ
# plt.xlabel('Patient Age',fontsize=14, fontweight='bold', color='gray')
# plt.ylabel('Frequency', fontsize=14, fontweight='bold',color='gray')
# plt.title('Distribution of Patient Age', fontsize=14, fontweight='bold',color='gray')

# # plt.ylim(0,500)
# plt.xlim(0,100)


# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # แสดงกราฟ
# plt.show()



df['patient_age'] = np.log1p(df['patient_age'])

print(df['patient_age'].head(1000))

df.to_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_log.csv",encoding="utf-8")


# # กำหนดขนาดกราฟ
# plt.figure(figsize=(10, 6))

# # สร้าง histogram ของข้อมูล
# sns.histplot(data=df, x='patient_age', kde=True, color='skyblue', bins=10)

# # เพิ่มชื่อแกน x, y และชื่อกราฟ
# plt.xlabel('Patient Age',fontsize=14, fontweight='bold', color='gray')
# plt.ylabel('Frequency', fontsize=14, fontweight='bold',color='gray')
# plt.title('Distribution of Patient Age', fontsize=14, fontweight='bold',color='gray')

# # plt.ylim(0,500)
# plt.xlim(0,5)


# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # แสดงกราฟ
# plt.show()


######################################################## -- Temp-- ############################################################

# scaler = MinMaxScaler()
# scaler_2 = StandardScaler()

# df['temperature_value'] = scaler.fit_transform(df[['temperature_value']])
# # df['temperature_value'] = scaler_2.fit_transform(df[['temperature_value']])

# # df['temperature_value'] = np.log1p(df['temperature_value'])
# print(df['temperature_value'].head(1000))


# # กำหนดขนาดกราฟ
# plt.figure(figsize=(10, 6))

# # สร้าง histogram ของข้อมูล
# sns.histplot(data=df, x='temperature_value', kde=True, color='skyblue', bins=10)

# # เพิ่มชื่อแกน x, y และชื่อกราฟ
# plt.xlabel('temperature_value',fontsize=14, fontweight='bold', color='gray')
# plt.ylabel('Frequency', fontsize=14, fontweight='bold',color='gray')
# plt.title('Distribution of temp', fontsize=14, fontweight='bold',color='gray')

# # plt.ylim(0,500)
# plt.xlim(0,1)


# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # แสดงกราฟ
# plt.show()