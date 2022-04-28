import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import nltk
import sklearn


def transform_text(text):
    text = text.lower()  # lower casing
    text = nltk.word_tokenize(text)  # tokenization

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()  # remove special characters

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # stopword & punctuation remove
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # stemming

    return " ".join(y)


tfidf = pickle.load(open('vectorized.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the Message")

if st.button("Predict"):
    # 1.Preprocess
    preprocess_sms = transform_text(input_sms)
    # 2.vectorized
    vector_input = tfidf.transform([preprocess_sms]).toarray()
    # 3.model
    result = model.predict(vector_input)[0]
    # 4.Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")