# Mail Spam Classifier 📧🚫

A machine learning-based email and SMS spam detection system that uses Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm to classify messages with high precision.

## 🚀 Performance
* **Accuracy:** ~98% on test data.
* **Algorithm:** Multinomial Naive Bayes.
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) with a 3,000-feature limit.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Libraries:**
    * `pandas` for data manipulation.
    * `nltk` for text preprocessing (tokenization, stopwords).
    * `scikit-learn` for machine learning and vectorization.
    * `joblib` for model persistence.

## 🧹 Preprocessing Pipeline
To achieve high accuracy, every message undergoes the following cleaning steps:
1. **Lowercasing:** Converts all text to lowercase.
2. **Tokenization:** Breaks sentences into individual words.
3. **Alpha-numeric Filtering:** Removes special characters and symbols.
4. **Stopword Removal:** Eliminates common words (e.g., "the", "is") that don't add predictive value.
5. **Stemming:** Reduces words to their root form (e.g., "joking" to "joke") using the Porter Stemmer.

## 📦 How to Use
### 1. Local Prediction
If you have downloaded the `spam_model.pkl` and `vectorizer.pkl` files, you can predict a single email:

```python
import joblib

model = joblib.load('spam_model.pkl')
tfidf = joblib.load('vectorizer.pkl')

# Example message
email = ["Free entry in 2 a wkly comp to win FA Cup final tkt 21st May 2005."]
vectorized = tfidf.transform(email)
prediction = model.predict(vectorized)
print(f"Result: {prediction[0]}")
