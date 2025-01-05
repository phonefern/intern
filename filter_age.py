import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp.csv",encoding="utf-8")
pd.set_option("display.max_rows",None)

df_filtered = df[(df['patient_age'] > 16) & (df['patient_age'] < 152)]

df_filtered.to_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age.csv",index=False)

age_count = df_filtered['patient_age'].value_counts().reset_index()


age_count.columns = ['age', 'count']

age_count = age_count.sort_values(by='age', ascending=True)

print(age_count)

symptoms_count = df_filtered.groupby('patient_age')[['pregnancy','temperature_value','fever', 'snot', 'sore_throat', 'phlegm', 'stomach_ache', 'Diarrhea', 'vomit', 'headache',
                      'itching', 'wound', 'rash', 'edema', 'ear_pain', 'fall', 'leg_pain', 'tumor','covid_positive',
                      'bite', 'tooth','tired','urine','heart','eye','crash','body','vagina','breast','blood']].sum()

print(symptoms_count)

# symptoms_count.to_csv("D:\project_kiosk\csv_plot_file\sort_age_symtoms.csv",index=True,encoding="utf-8")


symptoms_count_0_16 = symptoms_count.loc[0:16]
symptoms_count_17_49 = symptoms_count.loc[17:49]
symptoms_count_50_up = symptoms_count.loc[50:]
# 
# print(symptoms_count_0_16)
print(symptoms_count_17_49)
print(symptoms_count_50_up)

# correlation_matrix = symptoms_count_0_16.corr()
# correlation_matrix = symptoms_count_17_49.corr()
# correlation_matrix = symptoms_count_50_up.corr()

# # print(correlation_matrix)
# plt.figure(figsize=(8, 6))
# sns.heatmap(symptoms_count_0_16.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title(f'Correlation Plot of Symptoms for Age Group 0-16:')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.heatmap(symptoms_count_17_49.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title(f'Correlation Plot of Symptoms for Age Group 17-49:')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.heatmap(symptoms_count_50_up.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title(f'Correlation Plot of Symptoms for Age Group 50_up')
# plt.show()

# print(group_data.corr())

symptoms_count_0_16_sum = symptoms_count_0_16.sum()
symptoms_count_17_49_sum = symptoms_count_17_49.sum()
symptoms_count_50_up_sum = symptoms_count_50_up.sum()


plt.figure(figsize= (10,6))
barWidth = 0.25

bars1 = [symptoms_count_0_16_sum[col] for col in symptoms_count_0_16.columns]
bars2 = [symptoms_count_17_49_sum[col] for col in symptoms_count_17_49.columns]
bars3 = [symptoms_count_50_up_sum[col] for col in symptoms_count_50_up.columns]

r1 = range(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, bars1, color='pink', width=barWidth, edgecolor='grey', label='0-16 Years Old')
plt.bar(r2, bars2, color='black', width=barWidth, edgecolor='grey', label='17-49 Years Old')
plt.bar(r3, bars3, color='silver', width=barWidth, edgecolor='grey', label='50+ Years Old')

plt.xlabel('Symptoms', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], symptoms_count_0_16.columns, rotation=90)
plt.ylabel('Count', fontweight='bold')
plt.title('Contribution of Symtoms by Age Group', fontweight='bold')
plt.legend()
plt.show()








