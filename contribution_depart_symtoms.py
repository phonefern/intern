import pandas as pd
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age.csv", encoding="ISO-8859-1")

class_names = ['Emergency & Accident Unit','Heart Clinic','Neuro Med Center','OPD:EYE','Dental','OPD:MED','OPD:ENT','OPD:OBG','OPD:Surgery + Uro.','Orthopedic Surgery','GI Clinic','Breast Clinic',
                'Skin & Dermatology']

# หาจำนวนผู้ป่วยที่มีอาการไข้ในแต่ละแผนก
fever_count_by_department = df.groupby('target_department')['crash'].sum()

plt.figure(figsize=(10, 6))
fever_count_by_department.plot(kind='bar', color='black')
plt.title('crash')
plt.xlabel('Department')
plt.ylabel('Number of crash Cases')
plt.xticks(range(len(class_names)), class_names, rotation=30, ha='right')
plt.show()