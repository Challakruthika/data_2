import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load the labeled data
data = pd.read_csv('categorized_transactions.csv')

# Preprocessing: clean the description text
data['Description'] = data['Description'].str.lower().str.strip()
data['Category'] = data['Category'].str.strip()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['Description'], 
    data['Category'], 
    test_size=0.4, 
    random_state=42, 
    stratify=data['Category'] # Ensures balanced categories in train/test sets
)

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform the training data, then transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Model Performance on Test Set:")
print(classification_report(y_test, y_pred))

# Save the trained model and the vectorizer to disk
with open('category_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer have been saved to 'category_model.pkl' and 'vectorizer.pkl'")

# Example prediction
# To make a prediction on new data, you would load the model and vectorizer:
# with open('category_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)
# with open('vectorizer.pkl', 'rb') as f:
#     loaded_vectorizer = pickle.load(f)
# new_transaction = ["dinner with friends at a restaurant"]
# new_transaction_tfidf = loaded_vectorizer.transform(new_transaction)
# prediction = loaded_model.predict(new_transaction_tfidf)
# print(f"\nExample prediction for '{new_transaction[0]}': {prediction[0]}") 