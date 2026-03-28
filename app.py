
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Eco Friendly Leaf Cutlery - Data Analytics Dashboard")

uploaded = st.file_uploader("Upload dataset (or use default)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("leaf_cutlery_dataset.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Descriptive
st.subheader("Age Distribution")
fig = px.histogram(df, x="Age")
st.plotly_chart(fig)

st.subheader("Product Preference")
fig2 = px.bar(df["Preferred_Product"].value_counts())
st.plotly_chart(fig2)

# Preprocessing
df_ml = df.copy()

income_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df_ml["Income"] = income_encoder.fit_transform(df_ml["Income"])
df_ml["Preferred_Product"] = product_encoder.fit_transform(df_ml["Preferred_Product"])

X = df_ml.drop("Purchase_Intent", axis=1)
y = df_ml["Purchase_Intent"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Classification
model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

st.subheader("Classification Metrics")
st.write("Accuracy:",acc)
st.write("Precision:",prec)
st.write("Recall:",rec)
st.write("F1 Score:",f1)

probs = model.predict_proba(X_test)[:,1]
fpr,tpr,_ = roc_curve(y_test,probs)
roc_auc = auc(fpr,tpr)

roc_df = pd.DataFrame({"FPR":fpr,"TPR":tpr})
fig3 = px.line(roc_df,x="FPR",y="TPR",title="ROC Curve")
st.plotly_chart(fig3)
st.write("ROC AUC:",roc_auc)

feat = pd.Series(model.feature_importances_, index=X.columns)
fig4 = px.bar(feat.sort_values(), title="Feature Importance")
st.plotly_chart(fig4)

# Clustering
st.subheader("Customer Segmentation")
kmeans = KMeans(n_clusters=3, random_state=42)
df_ml["Cluster"] = kmeans.fit_predict(X)

fig5 = px.scatter(df_ml,x="Age",y="Monthly_Spend",color=df_ml["Cluster"].astype(str))
st.plotly_chart(fig5)

# Regression
st.subheader("Regression Example")
reg = LinearRegression()
reg.fit(X_train,y_train)
st.write("Regression model trained (demo).")

# Association Rules
st.subheader("Association Rules")

basket = pd.get_dummies(df["Preferred_Product"])
freq = apriori(basket,min_support=0.1,use_colnames=True)
rules = association_rules(freq,metric="confidence",min_threshold=0.3)

if not rules.empty:
    st.dataframe(rules[["antecedents","consequents","confidence","lift"]])
else:
    st.write("No strong rules found")

# New Customer Prediction
st.subheader("Predict New Customer Purchase Probability")

age = st.number_input("Age",18,70,30)
income = st.selectbox("Income",["Low","Medium","High"])
eco = st.slider("Eco Awareness",1,5,3)
event = st.slider("Event Frequency",0,10,2)
online = st.selectbox("Online Shopper",[0,1])

income_map = {"Low":0,"Medium":1,"High":2}
income_enc = income_map[income]

new = pd.DataFrame([[age,income_enc,eco,event,online,0,300]],
columns=["Age","Income","Eco_Awareness","Event_Frequency","Online_Shopper","Preferred_Product","Monthly_Spend"])

pred = model.predict_proba(new)[0][1]

st.success(f"Purchase Probability: {round(pred*100,2)}%")
