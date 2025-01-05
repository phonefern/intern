import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero.csv",encoding="utf-8")



label_encode = LabelEncoder()

df['target_department'] = label_encode.fit_transform(df['target_department'])

mapping = dict(zip(label_encode.classes_,label_encode.transform(label_encode.classes_)))


df.describe()
print(mapping)
print(df.head().to_string())

df.to_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode.csv",index=False,encoding="utf-8")

