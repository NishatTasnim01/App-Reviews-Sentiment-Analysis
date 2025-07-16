# Google PlayStore App Reviews Sentiment Analysis

This project performs sentiment analysis on user reviews of a popular Android application from the Google Play Store, specifically focusing on **Canva** app reviews. By leveraging NLP and ML techniques, we classify user sentiments as **Positive** or **Negative** based on their textual feedback.

## 📌 Project Summary

- **Dataset**: 1500 user reviews extracted from the Canva app on the Google Play Store.
- **Goal**: To clean, analyze, and build machine learning models that predict the sentiment (positive/negative) of app reviews.
- **Approach**:
  - Extensive text preprocessing.
  - EDA and visualization.
  - Feature engineering with Bag of Words (Binary, Count, and N-Gram models).
  - Logistic Regression classifier.
  - Accuracy comparison between feature extraction methods.

## 🧰 Technologies Used

- **Language**: Python 3
- **Libraries**: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `nltk` (tokenization, stopword removal, stemming)
  - `scikit-learn` (LogisticRegression, CountVectorizer, train_test_split)
  - `pickle` for model serialization

## 📂 Directory Structure

```

App-Reviews-Sentiment-Analysis/
│
├── App\_Reviews\_Sentiment\_Analysis.ipynb   ← Main Jupyter Notebook
├── data/
│   └── Canva\_reviews.xlsx                 ← Dataset (local)
├── Output/
│   ├── binary\_count\_vect.pkl              ← Pickled Binary Vectorizer
│   ├── binary\_count\_vect\_lr.pkl           ← Logistic Regression (Binary BoW)
│   ├── count\_vect.pkl                     ← Count Vectorizer
│   └── count\_vect\_lr.pkl                  ← Logistic Regression (Count BoW)
└── README.md

````

## 🔍 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Class distribution (Positive: 1032, Negative: 468 — **imbalanced**)
- Review score vs. sentiment correlation
- Text length and sentiment relationship visualized using `seaborn.displot`

### 2️⃣ Data Preprocessing
- Lowercasing, Tokenization (`nltk.word_tokenize`)
- Stopword and punctuation removal
- Stemming using `PorterStemmer`
- Vocabulary analysis (1720 unique tokens → reduced by `min_df=5`)

### 3️⃣ Feature Engineering
- **Binary Bag of Words**
- **Count Vectorization**
- **N-Gram (up to Trigram)** vectorization using `CountVectorizer(ngram_range=(1,3))`

### 4️⃣ Model Training
- **Logistic Regression** models trained for each vectorization type
- Performance evaluated on an 80/20 split

| Vectorizer Type | Train Accuracy | Test Accuracy |
|------------------|----------------|---------------|
| Binary BoW       | 96.08%         | 89.00%        |
| Count BoW        | 95.66%         | 88.33%        |
| N-Gram (1–3)     | ~97%+ (approx) | ~89%+         |

## 📊 Key Results

- Logistic Regression with Binary Bag-of-Words provided the best generalization with ~89% test accuracy.
- Adding n-grams increased vocabulary size and marginally improved performance.
- Stemmed vocabulary and filtered stopwords improved classifier performance and reduced noise.

## 🧪 Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/NishatTasnim01/App-Reviews-Sentiment-Analysis.git
   cd App-Reviews-Sentiment-Analysis

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook App_Reviews_Sentiment_Analysis.ipynb
   ```

## 💾 Model Saving & Deployment

* Trained vectorizers and models are serialized using `pickle`.
* Can be reloaded later for real-time review classification without retraining.

```python
with open("Output/count_vect.pkl", "rb") as f:
    vect = pickle.load(f)

with open("Output/count_vect_lr.pkl", "rb") as f:
    model = pickle.load(f)

# Predicting sentiment for a new review
X_new = vect.transform(["This app is fantastic and easy to use"])
pred = model.predict(X_new)
```

## 📝 License

This project is open-sourced under the MIT License.

## 👩‍💻 Developed By

**Nishat Tasnim**
[GitHub Profile](https://github.com/NishatTasnim01)

---

## ⭐ Show Your Support

If you found this useful, consider ⭐ starring the repository and contributing with pull requests or feedback!

