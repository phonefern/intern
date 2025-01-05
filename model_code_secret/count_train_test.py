import pandas as pd
from sklearn.model_selection import train_test_split

# Read data
df = pd.read_csv(r"./updated_visit_filter_depart_filter_zero_encode_range_temp_fil_age_filtered.csv")

# Feature selection
X = df[['fix_gender_id', 'patient_age', 'pregnancy', 'fever', 'cough', 'phlegm', 'Diarrhea', 'vomit', 'itching', 'wound', 'edema', 'tired', 'tumor', 'blood', 'covid_positive',
        'headache', 'eye', 'ear_pain', 'snot', 'tooth', 'sore_throat', 'breast', 'heart', 'rash', 'body', 'stomach_ache', 'vagina', 'urine', 'leg_pain',
        'fall', 'crash', 'bite']].copy()
y = df['target_department']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify symptom columns (excluding 'fix_gender_id' and 'patient_age')
symptom_columns = [col for col in X.columns if col not in ['fix_gender_id', 'patient_age']]

train_counts = X_train[symptom_columns].apply(lambda col: col.sum(), axis=0)

# Count the number of '1's in each symptom column for testing set
test_counts = X_test[symptom_columns].apply(lambda col: col.sum(), axis=0)

print("\nNumber of '1's in each symptom column for training set:")
print(train_counts)

print("\nNumber of '1's in each symptom column for testing set:")
print(test_counts)


# Count the number of samples in each set
train_count = len(X_train)
test_count = len(X_test)

print(f"Number of samples in the training set: {train_count}")
print(f"Number of samples in the testing set: {test_count}")

# Count the number of '1's (symptoms) in each row for the training set
train_symptom_counts = X_train[symptom_columns].sum(axis=1)

# Count the number of '1's (symptoms) in each row for the testing set
test_symptom_counts = X_test[symptom_columns].sum(axis=1)

# Count the occurrences of each count of symptoms in the training set
train_symptom_counts_distribution = train_symptom_counts.value_counts().sort_index()

# Count the occurrences of each count of symptoms in the testing set
test_symptom_counts_distribution = test_symptom_counts.value_counts().sort_index()

print("\nDistribution of number of symptoms in each row for the training set:")
print(train_symptom_counts_distribution)

print("\nDistribution of number of symptoms in each row for the testing set:")
print(test_symptom_counts_distribution)
