# -*- coding: utf-8 -*-
"""Plant_Survival.ipynb
# Will the Plant Survive ?
# Process the Data a bit
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("iotDataSet.csv")

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)
df.head()

def encode_categorical(df, column):
    values = df[column].values.tolist()
    lbl = LabelEncoder()
    lbl.fit(values)
    df.loc[:, column] = lbl.transform(df[column].values.tolist())
    encoder = {}
    encoder[column] = lbl
    joblib.dump(encoder, f"{column}_label_encoder.pkl")
    return df


print(df.columns)

"""# Encode data and build the Model"""
df = encode_categorical(df, "Sample Plant Dies ")
df.head()
X = df.drop('Sr. No.', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, df["Sample Plant Dies "], stratify=df["Sample Plant Dies "],
random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Score of Classifier:",clf.score(X_train, y_train))
predictions = clf.predict(X_test)
print("Predections for the test samples:",predictions)



print("Accuracy of Model is:",accuracy_score(y_test, predictions))
