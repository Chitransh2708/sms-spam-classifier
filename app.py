import nltk
import os

NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)

nltk.data.path = [NLTK_DIR]

# ONLY ensure punkt (nothing else)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DIR)


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/sms Spam Classifier")

input_sms = st.text_area("Enter the message to classify:")
if st.button('Predict'):

#1, preprocess
    transform_sms= transform_text(input_sms)
#2, vectorize
    vector_input = tfidf.transform([transform_sms])
#3, predict
    result = model.predict(vector_input)[0]
#4, display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")