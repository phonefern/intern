import pandas as pd

sheet_path = r"D:\project_kiosk\ภาพสหกิจ\package-data-bhh.xlsx"

df = pd.read_excel(sheet_path)

print(df)

df['output:'] = df['output:'].str.replace('+', ' ')
df.to_excel(sheet_path, index=False)

