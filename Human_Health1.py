import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly_express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


st.title("DIABETES ANALYSIS")
st.markdown("## OVERVIEW")

#import my csv file
st.markdown("### FIRST TEN OBSERVATIONS")
df = pd.read_csv("Diabetes.csv")
st.write(df.head(10))

st.markdown("### LAST TEN OBSERVATIONS")
df = pd.read_csv("Diabetes.csv")
st.write(df.tail(10))

st.markdown("### DATA INFO")
AK = df.shape
st.write(AK)

st.markdown("### CORRELATION")
correlation = df.corr()
st.write(correlation)

st.markdown("### BLOOD PRESSURE")
st.write(df["BloodPressure"].describe())

st.markdown("### FIRST TEN BLOOD PRESSURE")
st.write(df["BloodPressure"].head(10))

#UNIVARIATE ANALYSIS
st.markdown("### UNIVARIATE ANALYSIS")
st.markdown("### BLOOD PRESSURE ANALYSIS")
df = pd.read_csv("Diabetes.csv")
st.write(df["BloodPressure"].describe())

st.markdown("### BODY MASS INDEX ANALYSIS")
df = pd.read_csv("Diabetes.csv")
st.write(df["BMI"].describe())

st.markdown("### PREGNANCIES ANALYSIS")
df = pd.read_csv("Diabetes.csv")
st.write(df["Pregnancies"].describe())

st.markdown("### SKIN THICKNESS ANALYSIS")
df = pd.read_csv("Diabetes.csv")
st.write(df["SkinThickness"].describe())

st.markdown("### GLUCOSE ANALYSIS")
df = pd.read_csv("Diabetes.csv")
st.write(df["Glucose"].describe())

st.markdown("### INSULIN ANALYSIS")
df = pd.read_csv("Diabetes.csv")
st.write(df["Insulin"].describe())

st.markdown("### HISTOGRAM REPRESENTATION FOR BP")
BP = px.histogram(df["BloodPressure"], y= "BloodPressure", title="Pressure Distribution")
st.plotly_chart(BP, use_container_width=True)

st.markdown("### LINE GRAPH REPRESENTATION FOR BP")
BP = px.line(df["BloodPressure"], y= "BloodPressure", title="Pressure Distribution")
st.plotly_chart(BP, use_container_width=True)

st.markdown("### BAR REPRESENTATION FOR BP")
BP2 = px.bar(df["BloodPressure"], y= "BloodPressure", title="Pressure Distribution")
st.plotly_chart(BP2, use_container_width=True)

st.markdown("### HISTOGRAM REPRESENTATION FOR PREGNANCIES")
Pregg = px.histogram(df["Pregnancies"], y ="Pregnancies", title = "Pregnancies Distribution")
st.plotly_chart(Pregg, use_container_width = True)

st.markdown("### LINE GRAPH REPRESENTATION FOR PREGNANCIES")
Pregg = px.line(df["Pregnancies"], y ="Pregnancies", title = "Pregnancies Distribution")
st.plotly_chart(Pregg, use_container_width = True)

st.markdown("### BAR REPRESENTATION FOR PREGNANCIES")
Pregg = px.bar(df["Pregnancies"], y ="Pregnancies", title = "Pregnancies Distribution")
st.plotly_chart(Pregg, use_container_width = True)

#BIVARIATE ANALYSIS
st.markdown("## BIVARIATE ANALYSIS")
st.markdown("### Blood Pressure vs Pregnancies")
df2 = pd.DataFrame(df["BloodPressure"],df["Pregnancies"])
st.write(df2)

st.markdown("### Blood Pressure vs BMI")
df3 = pd.DataFrame(df["BloodPressure"],df["BMI"])
st.write(df3)

st.markdown("### Glucose vs Pregnancies")
df4 = pd.DataFrame(df["Glucose"],df["Pregnancies"])
st.write(df4)

st.markdown("### Skin Thickness vs Pregnancies")
df5 = pd.DataFrame(df["SkinThickness"],df["Pregnancies"])
st.write(df5)

st.markdown("### Age vs Pregnancies")
df6 = pd.DataFrame(df["Age"],df["Pregnancies"])
st.write(df6)

st.markdown("### Pregnancies vs Insulin")
df_ = pd.DataFrame(df["Pregnancies"],df["Insulin"])
st.write(df_)

st.markdown("# PREDICTIVE ANALYSIS")
X = df.drop("Outcome", axis=1)
Y = df["Outcome"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,Y_train) #training the model

st.markdown("## Outcome Prediction")
prediction = model.predict(X_test)
st.write(prediction)

st.markdown("## Model Evaluation")
accuracy = accuracy_score(prediction, Y_test)
st.write(accuracy)


#download by typing "python -m pip install scikit-learn"
#download by typing "python -m pip install matlib"
#download by typing "python -m pip install seaborn"