import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_save():
    df = pd.read_csv("categorized_transactions.csv")
    X = df["Description"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    print("✅ Model Accuracy:", pipeline.score(X_test, y_test))

    joblib.dump(pipeline, "category_model.pkl")
    print("✅ Model saved as 'category_model.pkl'")

if __name__ == "__main__":
    train_and_save()
