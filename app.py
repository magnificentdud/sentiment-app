import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@st.cache_resource
def train_model():
    df = pd.read_csv('data.csv')
    df['clean_review'] = df['review'].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['sentiment'], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return model, vectorizer

st.title("🎭 Sentiment Analyser")
st.write("Type any movie review and I'll predict if it's positive or negative.")

model, vectorizer = train_model()

text = st.text_area("Enter a review:", height=150)

if st.button("Analyse"):
    if text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max() * 100

        if prediction == "positive":
            st.success(f"Positive 😊 — {confidence:.1f}% confidence")
        else:
            st.error(f"Negative 😞 — {confidence:.1f}% confidence")
