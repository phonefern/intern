import pandas as pd
import re
df = pd.read_csv(r"D:\project_kiosk\visit_clean_temp_age_preg.csv",encoding="utf-8")
pd.set_option("display.max_rows",None)




from rich.console import Console
from rich.table import Table
console = Console()

all_tables = []
print("------------------------------------------------------------------------------------------------------------------------")
print("[Starting Count....]")
print('')
print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------ไข้ SELECT-----------------------------------------
df['fever'] = df['special_care_note'].apply(lambda x: 1 if 'ไข้' in str(x) else 0)

count_fever = df['fever'].value_counts()

fever_rows = df[df['fever'] == 1]
target_department_fever = fever_rows['target_department'].value_counts()

table_fever = Table(title="จำนวนอาการไข้")
table_fever.add_column("fever",style='cyan')
table_fever.add_column("counts",style='red3')

for text, count in count_fever.items():
    table_fever.add_row(str(text), str(count))


# print(target_department_fever)
console.print(table_fever)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------ไอ SELECT-----------------------------------------
df['cough'] = df['special_care_note'].apply(lambda x: 1 if ('ไอ' in str(x)) or ('เสียงเเหบ' in str(x)) or ('จาม' in str(x)) else 0)

count_cough = df['cough'].value_counts()

cough_rows = df[df['cough'] == 1]
target_department_cough = cough_rows['target_department'].value_counts()

table_cough = Table(title="จำนวนอาการไอ")
table_cough.add_column("cough",style='cyan')
table_cough.add_column("counts",style='red3')

for text, count in count_cough.items():
    table_cough.add_row(str(text), str(count))


# print('cough =',target_department_cough)
console.print(table_cough)

# target_department_cough.to_csv(r"D:\project_kiosk\count_depart_F\cough_count.csv",header=True)
print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------น้ำมูก SELECT-----------------------------------------
df['snot'] = df['special_care_note'].apply(lambda x: 1 if ('น้ำมูก' in str(x)) or ('คัดจมูก' in str(x)) or ('จมูกไม่ได้กลิ่น' in str(x)) else 0)

count_snot= df['snot'].value_counts()

snot_rows = df[df['snot'] == 1]
target_department_snot = snot_rows['target_department'].value_counts()

table_snot = Table(title="จำนวนผู้ป่วยมีน้ำมูก")
table_snot.add_column("snot",style='cyan')
table_snot.add_column("counts",style='red3')

for text, count in count_snot.items():
    table_snot.add_row(str(text), str(count))



console.print(table_snot)


print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------เจ็บคอ SELECT-----------------------------------------
df['sore_throat'] = df['special_care_note'].apply(lambda x: 1 if ('เจ็บคอ' in str(x)) or ('ระคายคอ' in str(x)) or ('คันคอ' in str(x)) or ('เเสบคอ' in str(x))  else 0)

count_sore_throat = df['sore_throat'].value_counts()

sore_throat_rows = df[df['sore_throat'] == 1]
target_department_sore_throat = sore_throat_rows['target_department'].value_counts()

table_sore_throat = Table(title="จำนวนอาการเจ็บคอ")
table_sore_throat.add_column("sore_throat",style='cyan')
table_sore_throat.add_column("counts",style='red3')

for text, count in count_sore_throat.items():
    table_sore_throat.add_row(str(text), str(count))



console.print(table_sore_throat)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------เสมหะ SELECT-----------------------------------------
df['phlegm'] = df['special_care_note'].apply(lambda x: 1 if 'เสมหะ' in str(x) else 0)

count_sore_phlegm = df['phlegm'].value_counts()


phlegm_rows = df[df['phlegm'] == 1]
target_department_phlegm = phlegm_rows['target_department'].value_counts()

table_sore_phlegm = Table(title="จำนวนอาการเสมหะ")
table_sore_phlegm.add_column("phlegm/เสมหะ",style='cyan')
table_sore_phlegm.add_column("counts",style='red3')

for text, count in count_sore_phlegm.items():
    table_sore_phlegm.add_row(str(text), str(count))



console.print(table_sore_phlegm)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------ปวดท้อง SELECT-----------------------------------------
df['stomach_ache'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'(ปวดท้อง|ท้องอืด|แน่นท้อง|ท้องผูก|กรดไหลย้อน|อาหารไม่ย่อย|กระเพาะ|เเน่นลิ้นปี่|ท้องเสีย|จุกเสียด|แสบท้อง|มวนท้อง|เรอบ่อย)', str(x)) else 0)

count_stomach_ache = df['stomach_ache'].value_counts()

stomach_ache_rows = df[df['stomach_ache'] == 1]
target_department_stomach_ache = stomach_ache_rows['target_department'].value_counts()

table_stomach_ache = Table(title="จำนวนอาการปวดท้อง")
table_stomach_ache.add_column("stomach_ache",style='cyan')
table_stomach_ache.add_column("counts",style='red3')

for text, count in count_stomach_ache.items():
    table_stomach_ache.add_row(str(text), str(count))

console.print(table_stomach_ache)
print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------ถ่ายเหลว SELECT-----------------------------------------
df['Diarrhea'] = df['special_care_note'].apply(lambda x: 1 if ('ถ่ายเหลว' in str(x)) or ('ท้องเสีย' in str(x)) else 0)

count_Diarrhea = df['Diarrhea'].value_counts()

Diarrhea_rows = df[df['Diarrhea'] == 1]
target_department_Diarrhea = Diarrhea_rows['target_department'].value_counts()

table_Diarrhea = Table(title="จำนวนอาการถ่ายเหลว")
table_Diarrhea.add_column("Diarrhea",style='cyan')
table_Diarrhea.add_column("counts",style='red3')

for text, count in count_Diarrhea.items():
    table_Diarrhea.add_row(str(text), str(count))



console.print(table_Diarrhea)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------อาเจียน SELECT-----------------------------------------
df['vomit'] = df['special_care_note'].apply(lambda x: 1 if ('อาเจียน' in str(x)) or ('คลื่นไส้' in str(x )) else 0)

count_vomit = df['vomit'].value_counts()

vomit_rows = df[df['vomit'] == 1]
target_department_vomit = vomit_rows['target_department'].value_counts()

table_vomit = Table(title="จำนวนอาการอาเจียน")
table_vomit.add_column("vomit",style='cyan')
table_vomit.add_column("counts",style='red3')

for text, count in count_vomit.items():
    table_vomit.add_row(str(text), str(count))



console.print(table_vomit)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------ปวดหัว SELECT-----------------------------------------
df['headache'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'(ปวดศีรษะ|ปวดหัว|เวียนศีรษะ|มึนศีรษะ|ไมเกรน|ปวดขมับ|บ้านหมุน|สมอง|มึน|วิงเวียน)', str(x)) else 0)

count_headache = df['headache'].value_counts()

headache_rows = df[df['headache'] == 1]
target_department_headache = headache_rows['target_department'].value_counts()

table_headache = Table(title="จำนวนอาการปวดหัว")
table_headache.add_column("headache",style='cyan')
table_headache.add_column("counts",style='red3')

for text, count in count_headache.items():
    table_headache.add_row(str(text), str(count))



console.print(table_headache)

# ----------------------------------คัน SELECT-----------------------------------------
df['itching'] = df['special_care_note'].apply(lambda x: 1 if 'คัน' in str(x) else 0)

count_itching = df['itching'].value_counts()

itching_rows = df[df['itching'] == 1]
target_department_itching = itching_rows['target_department'].value_counts()

table_itching = Table(title="จำนวนอาการคัน")
table_itching.add_column("itching",style='cyan')
table_itching.add_column("counts",style='red3')

for text, count in count_itching.items():
    table_itching.add_row(str(text), str(count))



console.print(table_itching)


print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------เเผล SELECT-----------------------------------------
df['wound'] = df['special_care_note'].apply(lambda x: 1 if ('แผล' in str(x)) or ('ฟกช้ำ' in str(x)) else 0)

count_wound = df['wound'].value_counts()

wound_rows = df[df['wound'] == 1]
target_department_wound = wound_rows['target_department'].value_counts()

table_wound = Table(title="จำนวนผู้ป่วยมีเเผล")
table_wound.add_column("wound",style='cyan')
table_wound.add_column("counts",style='red3')

for text, count in count_wound.items():
    table_wound.add_row(str(text), str(count))



console.print(table_wound)

print("------------------------------------------------------------------------------------------------------------------------")



#--------------------------------มีผื่น SELECT---------------------------------------------

df['rash'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'ผื่น|เชื้อรา|สิวผิว|ผมร่วง|ผิวเเห้ง|รังแค|รักแร้|งูสวัด' ,str(x)) else 0)

count_rash = df['rash'].value_counts()

rash_rows = df[df['rash'] == 1]
target_department_rash = rash_rows['target_department'].value_counts()

table_rash = Table(title="จำนวนผูัป่วยมีอาการผื่น")
table_rash.add_column("rash",style='cyan')
table_rash.add_column("counts",style='red3')



for text, count in count_rash.items():
    table_rash.add_row(str(text), str(count))


console.print(table_rash)

print("------------------------------------------------------------------------------------------------------------------------")

#--------------------------------มีอาการบวม SELECT---------------------------------------------

df['edema'] = df['special_care_note'].apply(lambda x: 1 if 'บวม' in str(x) else 0)

count_edema = df['edema'].value_counts()

edema_rows = df[df['edema'] == 1]
target_department_edema= edema_rows['target_department'].value_counts()

table_edema = Table(title="จำนวนผูัป่วยมีอาการบวม")
table_edema.add_column("edema",style='cyan')
table_edema.add_column("counts",style='red3')



for text, count in count_edema.items():
    table_edema.add_row(str(text), str(count))


console.print(table_edema)

print("------------------------------------------------------------------------------------------------------------------------")

#--------------------------------เจ็บหู SELECT---------------------------------------------

df['ear_pain'] = df['special_care_note'].apply(lambda x: 1 if 'หู' in str(x) else 0)

count_ear = df['ear_pain'].value_counts()

ear_pain_rows = df[df['ear_pain'] == 1]
target_department_ear_pain= ear_pain_rows['target_department'].value_counts()

table_ear = Table(title="จำนวนอาการเจ็บหู")
table_ear.add_column("ear_pain",style='cyan')
table_ear.add_column("counts",style='red3')

for text, count in count_ear.items():
    table_ear.add_row(str(text), str(count))


console.print(table_ear)

print("------------------------------------------------------------------------------------------------------------------------")


#--------------------------------COVID/ATK+ SELECT---------------------------------------------

df['covid_positive'] = df['special_care_note'].apply(lambda x: 1 if ('ATK+' in str(x)) or ('ATK +' in str(x)) else 0)

count_covid = df['covid_positive'].value_counts()

covid_positive_rows = df[df['covid_positive'] == 1]
target_department_covid_positive = covid_positive_rows['target_department'].value_counts()

table_covid = Table(title="ATK+")
table_covid.add_column("covid_positive",style='cyan')
table_covid.add_column("counts",style='red3')

for text, count in count_covid.items():
    table_covid.add_row(str(text), str(count))


console.print(table_covid)

print("------------------------------------------------------------------------------------------------------------------------")



#--------------------------------Fall SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['fall'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'ล้ม|ตก|เดินเซ' ,str(x)) else 0)

count_fall= df['fall'].value_counts()

fall_rows = df[df['fall'] == 1]
target_department_fall= fall_rows['target_department'].value_counts()

table_fall = Table(title="จำนวนคนล้ม/ตก")
table_fall.add_column("fall",style='cyan')
table_fall.add_column("counts",style='red3')

for text, count in count_fall.items():
    table_fall.add_row(str(text), str(count))


console.print(table_fall)

print("------------------------------------------------------------------------------------------------------------------------")



#--------------------------------leg_pain SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['leg_pain'] = df['special_care_note'].apply(lambda x: 1 if ('เจ็บขา' in str(x)) or ('ปวดขา' in str(x)) or ('ปวดเข่า' in str(x)) or ('เท้า' in str(x)) or ('ข้อเท้า' in str(x)) or ('เเพลง' in str(x)) else 0)

count_leg_pain= df['leg_pain'].value_counts()

leg_pain_rows = df[df['leg_pain'] == 1]
target_department_leg_pain = leg_pain_rows['target_department'].value_counts()  

table_leg_pain = Table(title="จำนวนผู้ป่วยปวดขา")
table_leg_pain.add_column("leg_pain",style='cyan')
table_leg_pain.add_column("counts",style='red3')

for text, count in count_leg_pain.items():
    table_leg_pain.add_row(str(text), str(count))


console.print(table_leg_pain)

print("------------------------------------------------------------------------------------------------------------------------")




# ----------------------------------มีก้อน SELECT-----------------------------------------
df['tumor'] = df['special_care_note'].apply(lambda x: 1 if ('ก้อน' in str(x)) or ('ตุ่ม' in str(x)) or ('ติ่ง' in str(x)) or ('เม็ด' in str(x)) else 0)

count_tumor = df['tumor'].value_counts()

tumor_rows = df[df['tumor'] == 1]
target_department_tumor = tumor_rows['target_department'].value_counts()

table_tumor = Table(title="จำนวนผู้ป่วยมีก้อน")
table_tumor.add_column("tumor",style='cyan')
table_tumor.add_column("counts",style='red3')

for text, count in count_tumor.items():
    table_tumor.add_row(str(text), str(count))


# print(target_department_fever)
console.print(table_tumor)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------โดนกัด SELECT-----------------------------------------
df['bite'] = df['special_care_note'].apply(lambda x: 1 if ('กัด' in str(x)) or ('บาด' in str(x)) else 0)

count_bite = df['bite'].value_counts()

bite_rows = df[df['bite'] == 1]
target_department_bite = bite_rows['target_department'].value_counts()

table_bite= Table(title="จำนวนผู้ป่วยโดนกัด")
table_bite.add_column("bite",style='cyan')
table_bite.add_column("counts",style='red3')

for text, count in count_bite.items():
    table_bite.add_row(str(text), str(count))



console.print(table_bite)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------ฟัน SELECT-----------------------------------------
df['tooth'] = df['special_care_note'].apply(lambda x: 1 if ('ฟัน' in str(x)) or ('ขูดหินปูน' in str(x)) or ('เคลือบฟลูออไรด์' in str(x)) else 0)

count_tooth = df['tooth'].value_counts()

tooth_rows = df[df['tooth'] == 1]
target_department_tooth= tooth_rows['target_department'].value_counts()

table_tooth= Table(title="จำนวนผู้ป่วยทำฟัน")
table_tooth.add_column("tooth",style='cyan')
table_tooth.add_column("counts",style='red3')

for text, count in count_tooth.items():
    table_tooth.add_row(str(text), str(count))



console.print(table_tooth)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------เหนื่อย SELECT-----------------------------------------
df['tired'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'(เหนื่อย|หอบ|เมื่อย|อ่อนเพลีย|หน้ามืด|อ่อนแรง|เพลีย|ไม่มีแรง|เป็นลม|วูบ|หายใจขัด|หายใจไม่สะดวก|หายใจไม่อิ่ม|ปอด)', str(x)) else 0)

count_tired= df['tired'].value_counts()

tired_rows = df[df['bite'] == 1]
target_department_tired= tired_rows['target_department'].value_counts()

table_tired= Table(title="จำนวนผู้ป่วยเหนื่อย")
table_tired.add_column("tired",style='cyan')
table_tired.add_column("counts",style='red3')

for text, count in count_tired.items():
    table_tired.add_row(str(text), str(count))



console.print(table_tired)

print("------------------------------------------------------------------------------------------------------------------------")





# ----------------------------------ปัสสาวะ SELECT-----------------------------------------
df['urine'] = df['special_care_note'].apply(lambda x: 1 if ('ปัสสาวะ' in str(x)) else 0)

count_urine= df['urine'].value_counts()

urinerows = df[df['urine'] == 1]
target_department_urine= urinerows['target_department'].value_counts()

table_urine= Table(title="จำนวนอาการปัสสาวะ")
table_urine.add_column("urine",style='cyan')
table_urine.add_column("counts",style='red3')

for text, count in count_urine.items():
    table_urine.add_row(str(text), str(count))



console.print(table_urine)

print("------------------------------------------------------------------------------------------------------------------------")

# ----------------------------------หน้าอก CAG ใจสั่น ECHO SELECT-----------------------------------------
df['heart'] = df['special_care_note'].apply(lambda x: 1 if  re.search(r'ใจสั่น|ใจเต้น|ใจหวิว|ตรวจหัวใจ|ใจโต|หน้าอก|หัวใจบีบ|เจ็บหน้าอก|จุกแน่นหน้าอก|หัวใจ|ลิ้นปี่|จุกแน่นกลางอก', str(x)) else 0)

count_chest = df['heart'].value_counts()

chest_rows = df[df['heart'] == 1]
target_department_chest = chest_rows['target_department'].value_counts()

table_chest= Table(title="จำนวนผู้ป่วยอาการด้านหัวใจ")
table_chest.add_column("chest",style='cyan')
table_chest.add_column("counts",style='red3')

for text, count in count_chest.items():
    table_chest.add_row(str(text), str(count))



console.print(table_chest)

print("------------------------------------------------------------------------------------------------------------------------")

#--------------------------------ตา SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['eye'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'เจ็บตา|ตาแดง|เคืองตา|คันตา|ตรวจตา|ตาเจ็บ|เข้าตา|ตาบวม|ขี้ตา|ตากุ้งยิง|ตามัว|ตาซ้าย|ตาขวา|ตาพร่า|ทิ่มตา|น้ำตา|เบ้าตา|ตาแห้ง|ต้อ|ขี้ตา|สายตา|มองไม่ชัด|ตามัว|ตามัว',str(x)) else 0)

count_eye= df['eye'].value_counts()

eye_rows = df[df['eye'] == 1]
target_department_eye= eye_rows['target_department'].value_counts()

table_eye = Table(title="เจ็บตา")
table_eye.add_column("eye",style='cyan')
table_eye.add_column("counts",style='red3')

for text, count in count_eye.items():
    table_eye.add_row(str(text), str(count))


console.print(table_eye)

print("------------------------------------------------------------------------------------------------------------------------")

#--------------------------------ชน SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['crash'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'ชน|กระเเทก|สะดุด|พลิก',str(x)) else 0)

count_crash = df['crash'].value_counts()

crash_rows = df[df['crash'] == 1]
target_department_crash= crash_rows['target_department'].value_counts()

table_crash = Table(title="ชน/กระเเทก")
table_crash.add_column("eye",style='cyan')
table_crash.add_column("counts",style='red3')

for text, count in count_crash.items():
    table_crash.add_row(str(text), str(count))


console.print(table_crash)

print("------------------------------------------------------------------------------------------------------------------------")

#--------------------------------เจ็บร่างกาย SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['body'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'ปวดไหล่|ปวดเอว|ปวดต้นคอ|ปวดนิ้ว|ปวดหลัง|เจ็บเเขน|ปวดเเขน|ปวดสะโพก|ปวดเข่า|ปวดข้อมือ|ปวดสะบัก|ปวดสะบัก|ก้น|ปวดคอ|ข้อศอก|มือซ้าย|มือขวา|เจ็บหน้าอก|ปวดบ่า',str(x)) else 0)

count_body = df['body'].value_counts()

body_rows = df[df['body'] == 1]
target_department_body= body_rows['target_department'].value_counts()

table_body = Table(title="ร่างกาย")
table_body.add_column("body",style='cyan')
table_body.add_column("counts",style='red3')

for text, count in count_body.items():
    table_body.add_row(str(text), str(count))


console.print(table_body)

print("------------------------------------------------------------------------------------------------------------------------")


#--------------------------------ช่องคลอด SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['vagina'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'คลอด|อวัยวะเพศ|ท้องน้อย|มดลูก|ตกขาว|เเพ้ท้อง|ประจำเดือน|ตั้งครรภ์',str(x)) else 0)

count_vagina = df['vagina'].value_counts()

vagina_rows = df[df['vagina'] == 1]
target_department_vagina= vagina_rows['target_department'].value_counts()

table_vagina = Table(title="ช่องคลอด")
table_vagina.add_column("vagina",style='cyan')
table_vagina.add_column("counts",style='red3')

for text, count in count_vagina.items():
    table_vagina.add_row(str(text), str(count))


console.print(table_vagina)

print("------------------------------------------------------------------------------------------------------------------------")

#--------------------------------เต้านม SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['breast'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'เต้านม',str(x)) else 0)

count_breast = df['breast'].value_counts()

breast_rows = df[df['breast'] == 1]
target_department_breast= breast_rows['target_department'].value_counts()

table_breast = Table(title="เต้านม")
table_breast.add_column("breast",style='cyan')
table_breast.add_column("counts",style='red3')

for text, count in count_breast.items():
    table_breast.add_row(str(text), str(count))


console.print(table_breast)

print("------------------------------------------------------------------------------------------------------------------------")

#--------------------------------เลือด SELECT---------------------------------------------

# df['fall'] = df['special_care_note'].apply(lambda x: 1 if ('ล้ม' in str(x)) or ('ตก' in str(x)) else 0)
df['blood'] = df['special_care_note'].apply(lambda x: 1 if re.search(r'เลือด',str(x)) else 0)

count_blood = df['blood'].value_counts()

blood_rows = df[df['blood'] == 1]
target_department_blood = blood_rows['target_department'].value_counts()

table_blood = Table(title="เลือด")
table_blood.add_column("blood",style='cyan')
table_blood.add_column("counts",style='red3')

for text, count in count_blood.items():
    table_blood.add_row(str(text), str(count))


console.print(table_blood)

print("------------------------------------------------------------------------------------------------------------------------")

# สร้าง DataFrame เพื่อเก็บข้อมูลจากทุกตาราง
all_depart_counts = []

# เพิ่มข้อมูลจากตารางของอาการไข้
for text, count in target_department_fever.items():
    all_depart_counts.append({'symptom': 'fever', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการไอ
for text, count in target_department_cough.items():
    all_depart_counts.append({'symptom': 'cough', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการน้ำมูก
for text, count in target_department_snot.items():
    all_depart_counts.append({'symptom': 'snot', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการเจ็บคอ
for text, count in target_department_sore_throat.items():
    all_depart_counts.append({'symptom': 'sore_throat', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการเสมหะ
for text, count in target_department_phlegm.items():
    all_depart_counts.append({'symptom': 'phlegm', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการปวดท้อง
for text, count in target_department_stomach_ache.items():
    all_depart_counts.append({'symptom': 'stomach_ache', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการถ่ายเหลว
for text, count in target_department_Diarrhea.items():
    all_depart_counts.append({'symptom': 'Diarrhea', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการอาเจียน
for text, count in target_department_vomit.items():
    all_depart_counts.append({'symptom': 'vomit', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการปวดหัว
for text, count in target_department_headache.items():
    all_depart_counts.append({'symptom': 'headache', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการคัน
for text, count in target_department_itching.items():
    all_depart_counts.append({'symptom': 'itching', 'target_department': text, 'counts': count})


# เพิ่มข้อมูลจากตารางของอาการมีเเผล
for text, count in target_department_wound.items():
    all_depart_counts.append({'symptom': 'wound', 'target_department': text, 'counts': count})


# เพิ่มข้อมูลจากตารางของอาการมีก้อน
for text, count in target_department_tumor.items():
    all_depart_counts.append({'symptom': 'tumor', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการกัด
for text, count in target_department_bite.items():
    all_depart_counts.append({'symptom': 'bite', 'target_department': text, 'counts': count})


# เพิ่มข้อมูลจากตารางของอาการผื่น
for text, count in target_department_rash.items():
    all_depart_counts.append({'symptom': 'rash', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการบวม
for text, count in target_department_edema.items():
    all_depart_counts.append({'symptom': 'edema', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางของอาการเจ็บหู
for text, count in target_department_ear_pain.items():
    all_depart_counts.append({'symptom': 'ear_pain', 'target_department': text, 'counts': count})


# เพิ่มข้อมูลจากตารางของCOVID/ATK+
for text, count in target_department_covid_positive.items():
    all_depart_counts.append({'symptom': 'covid_positive', 'target_department': text, 'counts': count})


# เพิ่มข้อมูลจากตารางของจำนวนคนล้ม/ตก
for text, count in target_department_fall.items():
    all_depart_counts.append({'symptom': 'fall', 'target_department': text, 'counts': count})

# เพิ่มข้อมูลจากตารางจำนวนผู้ป่วยปวดขา
for text, count in target_department_leg_pain.items():
    all_depart_counts.append({'symptom': 'leg_pain', 'target_department': text, 'counts': count})



# สร้าง DataFrame จาก List ข้อมูล
all_department_data = pd.DataFrame(all_depart_counts, columns=['symptom', 'target_department', 'counts'])



# บันทึก DataFrame เป็นไฟล์ CSV
all_department_data.to_csv(r"D:\project_kiosk\new_world\visit_filter_depart_filter_zero_encode_range_temp_fil_age_filter_de.csv", index=False, encoding="utf-8")

# df.to_csv(r"D:\project_kiosk\visit_column_ok_v2.csv",index=False,encoding="utf-8")




