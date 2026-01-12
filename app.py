#Importing necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#Tf refers to Term Feature and TDF reffers to Inverse Document Feature
#Used to tokenize the word into numbers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix,classification_report
data=pd.read_csv("spam.csv",encoding='latin-1')
print("Sample data:",data.head(10))
x=data['v2'] #feature
y=data['v1'].map({'ham':0,'spam':1}) #target
#Splitting the dataset into two i.e.,train and test data
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
#converting words into tokens
vectorizer=TfidfVectorizer()
x_train_tdidf=vectorizer.fit_transform(x_train)
x_test_tdidf=vectorizer.transform(x_test)
#Using Logistic Regression Model
model1=LogisticRegression()
model1.fit(x_train_tdidf,y_train)
y_pred1=model1.predict(x_test_tdidf)

#Using Naive Bayes
model2=MultinomialNB()
model2.fit(x_train_tdidf,y_train)
y_pred2=model2.predict(x_test_tdidf)

#implementing using streamlit application
st.title("📧 Spam Email Classifier")
st.write("This app classifies emails/messages as **Spam** or **Ham (Not Spam)** using ML models.")

user_input = st.text_input("Enter an email/message:")
if st.button("Classify"):
  if user_input.strip()!="":
    user_input_tfidf=vectorizer.transform([user_input])
    prediction1=model1.predict(user_input_tfidf)[0]
    prediction2=model2.predict(user_input_tfidf)[0]
    st.subheader("Prediction Results")
    st.write("**Logistic Regression:**", "🚨 Spam" if prediction1==1 else "✅ Ham")
    st.write("**Naive Bayes:**", "🚨 Spam" if prediction2==1 else "✅ Ham")

#Displaying Model performance
st.subheader("Model Performance on Test Data")

st.write("### Logistic Regression")
st.write("Accuracy:", accuracy_score(y_test,y_pred1))
st.text("Classification Report:\n" + classification_report(y_test, y_pred1))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred1), annot=True, fmt='d', cmap="Blues", ax=ax)
ax.set_title("Logistic Regression Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.write("### Naive Bayes")
st.write("Accuracy:", accuracy_score(y_test, y_pred2))
st.text("Classification Report:\n" + classification_report(y_test,y_pred2))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred2), annot=True, fmt='d', cmap="Greens", ax=ax)
ax.set_title("Naive Bayes Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
