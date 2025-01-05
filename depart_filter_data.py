import pandas as pd

df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv", encoding="utf-8")

target_departments = [2001,3006, 3007, 3011, 3012, 3015, 3017,3018, 3020, 3023, 3054, 3080, 3172, 3421]
filtered_df = df[df['target_department'].isin(target_departments)]

# filtered_df = filtered_df[filtered_df['target_department'].isin(target_departments)]

# count_df = len(df)
# count_filtered_rows = len(filtered_df)
# count_unfiltered_rows = len(df) - count_filtered_rows                          

# print("จำนวน Dataframe ทั้งหมด :",count_df)
# print("จำนวน Dataframe หลังถูกกรอง :",count_filtered_rows)
# print("จำนวน Dataframe กรองไป :",count_unfiltered_rows)                  

filtered_df['target_department'].replace(3172,3015, inplace=True)

print(filtered_df)
filtered_df.to_csv(r"D:\project_kiosk\new_world\visit_filter_depart.csv", index=False)