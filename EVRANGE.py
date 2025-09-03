import pandas as pd

df = pd.read_csv("ev_range_prediction_dataset.csv")


print("Shape of dataset:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)


import numpy as np
df.loc[df.sample(frac=0.01, random_state=42).index, "Battery_Capacity_kWh"] = np.nan


for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)


for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


df.drop_duplicates(inplace=True)


from sklearn.preprocessing import LabelEncoder

cat_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = scaler.fit_transform(df[num_cols])


df.to_csv("ev_range_prediction_cleaned.csv", index=False)


from google.colab import files
files.download("ev_range_prediction_cleaned.csv")