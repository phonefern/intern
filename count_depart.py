import pandas as pd

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(r"D:\project_kiosk\csv_plot_file\visit_filter_depart_out.csv", encoding='utf-8')

# นับความถี่ของค่าในคอลัมน์ "target_department"
department_counts = df['target_department'].value_counts().sort_index()

# สร้าง DataFrame จาก Series ของค่าความถี่
department_counts_df = pd.DataFrame(department_counts)

class_names = ['Heart Clinic','Neuro Med Center','OPD:EYE','Dental','OPD:MED','OPD:ENT','OPD:OBG','OPD:PED','OPD:Surgery + Uro.','Orthopedic Surgery','GI Clinic','Breast Clinic'
               ,'Skin & Dermatology','Women Health Center']

print(department_counts_df)




# บันทึกเป็นไฟล์ CSV ใหม่
department_counts_df.to_csv(r"D:\project_kiosk\csv_plot_file\count_department_out_line.csv", encoding='utf-8')

print("บันทึกข้อมูลเรียบร้อยแล้ว")
