import pandas as pd

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age.csv", encoding="utf-8")

# กรองแถวที่มีค่า 0 ทั้งหมดในคอลัมทุกคอลัม
filtered_df = df[~df[['temperature_value','fever', 'snot', 'sore_throat', 'phlegm', 'stomach_ache', 'Diarrhea', 'vomit', 'headache',
                      'itching', 'wound', 'rash', 'edema', 'ear_pain', 'fall', 'leg_pain', 'tumor','covid_positive',
                      'bite', 'tooth','tired','urine','heart','eye','crash','body','vagina','breast','blood']].eq(0).all(axis=1)]

print(filtered_df)

# นับจำนวนแถวที่ถูกกรอง
filtered_row_count = len(filtered_df)
print(f"จำนวนแถวที่ถูกกรอง: {filtered_row_count}")

# filtered_df.to_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv", index=False, encoding="utf-8")