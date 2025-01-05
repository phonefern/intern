import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv(r"./updated_visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

# คำนวณค่า Correlation
corr = df.corr()

# สร้าง Heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
