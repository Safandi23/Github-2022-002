# =========================================
# STREAMLIT APP
# KLASIFIKASI SENTIMEN TEKS SEPAK BOLA
# DATASET: id | text | polarity
# FILE: train.csv
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# LOAD DATASET
# =========================================
@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset/train.csv")
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_dataset()

# =========================================
# TOKENIZER (SAMA DENGAN TRAINING)
# =========================================
import joblib

tokenizer = joblib.load("models/tokenizer.pkl")
MAX_LEN = 100




# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_models():
    return {
        "LSTM Base (Non-Pretrained)": tf.keras.models.load_model("models/lstm_base.h5"),
        "GloVe LSTM (Pretrained)": tf.keras.models.load_model("models/glove_lstm.h5"),
        "FastText LSTM (Pretrained)": tf.keras.models.load_model("models/fasttext_lstm.h5"),
    }

models = load_models()

# =========================================
# LABEL MAPPING
# =========================================
label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# =========================================
# STREAMLIT UI
# =========================================
st.title("⚽ Klasifikasi Sentimen Teks Sepak Bola")
st.write(
    "Aplikasi klasifikasi sentimen komentar sepak bola "
    "menggunakan Neural Network dan Transfer Learning."
)

model_choice = st.selectbox(
    "Pilih Model",
    list(models.keys())
)

text_input = st.text_area(
    "Masukkan teks komentar sepak bola:",
    placeholder="Contoh: Wasit sangat merugikan tim kami malam ini!"
)

# =========================================
# PREDIKSI
# =========================================
if st.button("Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        seq = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        prediction = models[model_choice].predict(padded)
        predicted_label = np.argmax(prediction, axis=1)[0]

        sentiment = label_map[predicted_label]

        st.subheader("Hasil Prediksi:")
        if sentiment == "Positive":
            st.success(f"Sentimen: **{sentiment}**")
        elif sentiment == "Neutral":
            st.info(f"Sentimen: **{sentiment}**")
        else:
            st.error(f"Sentimen: **{sentiment}**")
