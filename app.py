import streamlit as st
import joblib
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(page_title="Hate Speech Detector", page_icon="")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+|https?://\S+|[^a-z\s]', '', text)
    return text

@st.cache_resource
def load_ml_assets():
    model = joblib.load('models/logistic_regression_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return model, tfidf

@st.cache_resource
def load_dl_assets():
    model = load_model('models/lstm_hate_speech_model.h5')
    with open('models/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


st.title("Détecteur de Propos Haineux")
st.write("Entrez une phrase pour analyser si elle est haineuse, insultante ou neutre.")


option = st.selectbox("Choisissez le modèle d'analyse :", 
                     ("Machine Learning (Logistic Regression)", "Deep Learning (LSTM)"))

user_input = st.text_area("Saisissez votre texte ici :", placeholder="Tapez quelque chose...")

if st.button("Analyser"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer du texte.")
    else:
        cleaned_input = clean_text(user_input)
        labels = {0: "Hate Speech (Haine)", 1: "Offensive (Insultant)", 2: "Neither (Neutre)"}
        
        if option == "Machine Learning (Logistic Regression)":
            model_ml, tfidf = load_ml_assets()
            vec = tfidf.transform([cleaned_input])
            prediction = model_ml.predict(vec)[0]
            probs = model_ml.predict_proba(vec)[0]
            
        else: 
            model_dl, tokenizer = load_dl_assets()
            seq = tokenizer.texts_to_sequences([cleaned_input])
            padded = pad_sequences(seq, maxlen=50)
            pred_probs = model_dl.predict(padded)[0]
            prediction = np.argmax(pred_probs)
            probs = pred_probs

   
        res_label = labels[prediction]
        
        if prediction == 0:
            st.error(f"Résultat : {res_label}")
        elif prediction == 1:
            st.warning(f"Résultat : {res_label}")
        else:
            st.success(f"Résultat : {res_label}")
            
        st.write(f"**Confiance :** {max(probs)*100:.2f}%")
        
       
        st.bar_chart({labels[i]: probs[i] for i in range(len(labels))})