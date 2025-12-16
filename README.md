# Sentiment Analysis: Naive Bayes vs SVM (LinearSVC)

Proyek ini adalah implementasi Machine Learning untuk melakukan Analisis Sentimen pada dataset ulasan film IMDb (50.000 data). Proyek ini membandingkan kinerja antara algoritma probabilistik (**Naive Bayes**) dan algoritma berbasis margin (**Support Vector Machine**).

## Hasil Eksperimen

Berdasarkan pengujian pada 10.000 data uji (20% split), berikut adalah perbandingan kinerjanya:

| Model | Akurasi | Waktu Training | Kesimpulan |
|-------|---------|----------------|------------|
| **Naive Bayes** | 84.87% | Sangat Cepat | Baik sebagai baseline, namun cenderung memiliki False Positive lebih tinggi. |
| **SVM (Linear)**| 88.61% | Cepat | Memberikan akurasi terbaik dan generalisasi yang lebih kuat pada data teks dimensi tinggi. |

## Teknologi yang Digunakan

* **Bahasa:** Python 3.10+
* **Environment:** Virtual Environment (venv)
* **Library Utama:**
    * `scikit-learn`: Modeling (MultinomialNB, LinearSVC) & TF-IDF
    * `pandas` & `numpy`: Manipulasi Data
    * `nltk`: Preprocessing (Stopwords, Tokenization)
    * `seaborn` & `matplotlib`: Visualisasi Data

## Cara Menjalankan Project

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/son31dt/movie_review_nlp.git](https://github.com/son31dt/movie_review_nlp.git)
    cd movie_review_nlp
    ```

2.  **Setup Environment**
    ```bash
    python -m venv venv
    source venv/Scripts/activate  # Untuk Windows (Git Bash/Warp)
    # atau: venv\Scripts\activate # Untuk Windows (CMD)
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Notebook**
    Buka file `analisis_sentimen.ipynb` menggunakan VS Code atau Jupyter Notebook.

## Metodologi
1.  **Preprocessing:** Cleaning (Regex), Lowercasing.
2.  **Feature Extraction:** TF-IDF Vectorizer (Max features: 5000).
3.  **Modeling:** Komparasi Multinomial Naive Bayes vs LinearSVC.
4.  **Evaluasi:** Akurasi, Confusion Matrix, dan Waktu Eksekusi.