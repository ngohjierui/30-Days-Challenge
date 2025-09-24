# fake_news_classifier.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_data(filename: str) -> pd.DataFrame:
    """
    Load dataset from the 'data' folder.
    Falls back to current folder if not found.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))   # folder where script is
    data_path = os.path.join(base_dir, "data", filename)    # look inside /data/
    
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, filename)        # fallback: same folder
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Could not find {filename}. "
            f"Place it in a 'data/' folder or next to this script."
        )
    
    df = pd.read_csv(data_path)
    print("✅ Dataset loaded successfully!")
    print("Dataset Sample:\n", df.head(), "\n")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Prepare features (X) and labels (y).
    - Drop rows with missing labels
    - Fill NaN text with empty string
    """
    missing_labels = df["label"].isna().sum()
    if missing_labels > 0:
        print(f"⚠️ Dropping {missing_labels} rows with missing labels.")
        df = df.dropna(subset=["label"])
    
    X = df["text"].fillna("").astype(str)
    y = df["label"].astype(str)   # ensure labels are strings
    
    return X, y


def split_data(X, y):
    """
    Split dataset into train/test sets with stratification if possible.
    """
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train Naive Bayes model with TF-IDF and evaluate it.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)

    print(f"\n🎯 Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:\n", report)


if __name__ == "__main__":
    data = load_data("fake_news.csv")        # <-- just the file name
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    train_and_evaluate(X_train, X_test, y_train, y_test)
