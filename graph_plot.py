import pandas as pd
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(r"./model_code_secret\updated_visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv", encoding='utf-8')

class_names = ['Emergency & Accident Unit','Heart Clinic','Neuro Med Center','OPD:EYE','Dental','OPD:MED','OPD:ENT','OPD:OBG','OPD:Surgery + Uro.','Orthopedic Surgery','GI Clinic','Breast Clinic'
               ,'Skin & Dermatology']

class_names_gender = ['None','Men','Female']


# # นับความถี่ของค่าในคอลัมน์ "target_department"
department_counts = df['fix_gender_id'].value_counts().sort_index()
print(department_counts)



# สร้างกราฟแท่ง
plt.figure(figsize=(10, 6))
plt.bar(class_names_gender, department_counts, color='black')
plt.xlabel('fix_gender_id')
plt.ylabel('Frequency')
plt.title('Frequency of Gender')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# แสดงกราฟ
plt.show()

# บันทึกเป็นไฟล์ CSV ใหม่
# department_counts_df.to_csv(r"D:\project_kiosk\csv_plot_file\count_department.csv", encoding='utf-8')

print("บันทึกข้อมูลเรียบร้อยแล้ว")
####################################################################################################################################

# # กรองข้อมูลที่มีอายุไม่เกิน 113
# df_filtered = df[df['patient_age'] <= 113]

# # นับความถี่ของค่าในคอลัมน์ "fix_gender_id"
# department_counts = df_filtered['fix_gender_id'].value_counts().sort_index()
# print(department_counts)

# # สร้างกราฟแท่งสำหรับ fix_gender_id
# class_names_gender = ['None', 'Men', 'Female']
# plt.figure(figsize=(10, 6))
# plt.bar(class_names_gender, department_counts, color='black')
# plt.xlabel('fix_gender_id')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# # นับความถี่ของค่าในคอลัมน์ "patient_age"
# age_counts = df_filtered['patient_age'].value_counts().sort_index()
# print(age_counts)

# # คำนวณค่าทางสถิติ
# count = len(df_filtered['patient_age'])
# min_age = df_filtered['patient_age'].min()
# max_age = df_filtered['patient_age'].max()
# mean_age = df_filtered['patient_age'].mean()
# std_age = df_filtered['patient_age'].std()
# median_age = df_filtered['patient_age'].median()

# # สร้างกราฟแท่งสำหรับ age
# plt.figure(figsize=(10, 6))
# plt.bar(age_counts.index, age_counts.values, color='black')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.title('Frequency of Patient Age')

# # เพิ่มข้อมูลค่าทางสถิติบนกราฟ
# plt.text(0.95, 0.95, f'Count: {count}', ha='right', va='top', transform=plt.gca().transAxes)
# plt.text(0.95, 0.9, f'Min: {min_age}', ha='right', va='top', transform=plt.gca().transAxes)
# plt.text(0.95, 0.85, f'Max: {max_age}', ha='right', va='top', transform=plt.gca().transAxes)
# plt.text(0.95, 0.8, f'Mean: {mean_age:.2f}', ha='right', va='top', transform=plt.gca().transAxes)
# plt.text(0.95, 0.75, f'Std: {std_age:.2f}', ha='right', va='top', transform=plt.gca().transAxes)
# plt.text(0.95, 0.7, f'Median: {median_age}', ha='right', va='top', transform=plt.gca().transAxes)

# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # แสดงกราฟ
# plt.show()

# print("บันทึกข้อมูลเรียบร้อยแล้ว")

####################################################################################################################################


# class_names_gender = ['None','High_Temperature']


# # # นับความถี่ของค่าในคอลัมน์ "target_department"
# department_counts = df['temperature_value'].value_counts().sort_index()
# print(department_counts)



# # สร้างกราฟแท่ง
# plt.figure(figsize=(10, 6))
# plt.bar(class_names_gender, department_counts, color='black')
# plt.xlabel('temperature_value')
# plt.ylabel('Frequency')
# # plt.title('Frequency of Visits by Department')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

# # แสดงกราฟ
# plt.show()
