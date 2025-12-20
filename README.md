# 🎬 Movie Sentiment Analysis System (NLP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

A comprehensive Sentiment Analysis project comparing **Naive Bayes** and **Support Vector Machine (LinearSVC)** models on 50,000 IMDb movie reviews. This project includes a complete machine learning pipeline from data preprocessing to a deployable web application.

## 🚀 Live Demo

Check out the running application here:
👉 **https://sentimen-film-nlp.streamlit.app**

---

## 📊 Project Overview

This system classifies movie reviews as either **Positive** or **Negative**. It serves as a comparative study to determine which algorithm performs better in terms of accuracy and computational efficiency.

### Key Features:
* **Text Preprocessing:** Regex-based cleaning, lowercasing, and noise removal.
* **Feature Extraction:** TF-IDF Vectorization (Top 5,000 features).
* **Modeling:** Comparison between Probabilistic (Naive Bayes) and Geometric (SVM) approaches.
* **Web Interface:** Interactive frontend built with **Streamlit** for real-time sentiment prediction.

---

## 🏆 Model Performance

Based on the test set (20% split), here are the evaluation results:

| Model | Accuracy | Training Time | Prediction Time | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | 84.87% | ~0.05s | ~0.009s | Very fast, suitable for low-resource systems. |
| **SVM (Linear)**| **88.61%** | ~2.25s | ~0.007s | **Best Accuracy**, robust generalization. |

---

## 📂 Project Structure

```text
movie_review_nlp/
├── app.py                   # Main Streamlit application
├── analisis_sentimen.ipynb  # Jupyter Notebook for training & analysis
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
├── .gitignore               # Git configuration
└── models/                  # Saved serialized models (.pkl)
    ├── tfidf_vectorizer.pkl
    ├── nb_model.pkl
    └── svm_model.pkl
```


## 🛠️ Installation & Usage

If you want to run this project locally on your machine:

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/son31dt/movie_review_nlp.git](https://github.com/son31dt/movie_review_nlp.git)
    cd movie_review_nlp
    ```

2.  **Create a Virtual Environment**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment (Windows Git Bash/Warp):
    source venv/Scripts/activate

    # Activate the environment (Windows CMD):
    # venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```

---

## 🧠 Methodology (Data Flow)

1.  **Input:** Raw text data from IMDb Dataset.
2.  **Preprocessing:** Cleaning HTML tags and non-alphabetical characters using Regex.
3.  **Training:** TF-IDF transformation followed by model fitting (Naive Bayes & SVM).
4.  **Deployment:** The best-performing model is saved (`.pkl`) and loaded into the Streamlit app for inference.