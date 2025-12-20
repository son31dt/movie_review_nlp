import streamlit as st
import joblib
import re
import time

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen Film",
    page_icon="🎬",
    layout="centered"
)

# --- 2. FUNGSI LOAD MODEL (Dichace agar cepat) ---
@st.cache_resource
def load_models():
    # Load vectorizer dan model yang sudah disimpan
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    nb_model = joblib.load('nb_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    return tfidf, nb_model, svm_model

tfidf, nb_model, svm_model = load_models()

# --- 3. FUNGSI PREPROCESSING (Harus sama dengan di Notebook) ---
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)       # Hapus HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus non-huruf
    text = text.lower()                     # Lowercase
    return text

# --- 4. TAMPILAN UTAMA (UI) ---
st.title("🎬 Analisis Sentimen Ulasan Film")
st.write("Aplikasi ini menggunakan Machine Learning untuk mendeteksi apakah sebuah ulasan film bernada **Positif** atau **Negatif**.")

st.markdown("---")

# Input User
user_input = st.text_area("Masukkan Ulasan Film (Bahasa Inggris):", height=150, placeholder="Example: The movie was absolutely fantastic! I loved the acting.")

# Tombol Prediksi
if st.button("Analisis Sentimen 🚀"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        # 1. Preprocessing & Vectorization
        clean_input = clean_text(user_input)
        vector_input = tfidf.transform([clean_input])

        # 2. Prediksi
        # Naive Bayes
        start_nb = time.time()
        pred_nb = nb_model.predict(vector_input)[0]
        prob_nb = nb_model.predict_proba(vector_input).max()
        time_nb = time.time() - start_nb

        # SVM
        start_svm = time.time()
        pred_svm = svm_model.predict(vector_input)[0]
        # SVM LinearSVC tidak punya predict_proba secara default, kita pakai decision_function
        score_svm = svm_model.decision_function(vector_input)[0] 
        time_svm = time.time() - start_svm

        # Labeling
        label_map = {1: "POSITIF", 0: "NEGATIF"}
        color_map = {1: "green", 0: "red"}

        # 3. Tampilkan Hasil (Kolom Berdampingan)
        st.subheader("Hasil Prediksi")
        
        col1, col2 = st.columns(2)

        with col1:
            st.info("🤖 Naive Bayes")
            hasil_nb = label_map[pred_nb]
            st.markdown(f"<h3 style='color: {color_map[pred_nb]};'>{hasil_nb}</h3>", unsafe_allow_html=True)
            st.write(f"Confidence: {prob_nb:.2%}")
            st.write(f"Waktu: {time_nb:.4f} detik")

        with col2:
            st.success("⚔️ SVM (LinearSVC)")
            hasil_svm = label_map[pred_svm]
            st.markdown(f"<h3 style='color: {color_map[pred_svm]};'>{hasil_svm}</h3>", unsafe_allow_html=True)
            st.write(f"Margin Score: {score_svm:.2f}")
            st.write(f"Waktu: {time_svm:.4f} detik")
            
        # Kesimpulan sederhana
        if pred_nb == pred_svm:
            st.success(f"✅ Kedua model sepakat bahwa ulasan ini **{label_map[pred_nb]}**.")
        else:
            st.warning("⚠️ Model memiliki prediksi yang berbeda.")

# Footer
st.markdown("---")
st.caption("Dibuat dengan Python & Streamlit | Project Portofolio")