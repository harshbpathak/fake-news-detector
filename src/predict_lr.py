import joblib

# Load models and vectorizer
lr_model = joblib.load('./models/logistic_model.pkl')
rf_model = joblib.load('./models/random_forest_model.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

def predict_news(text):
    vectorized_input = vectorizer.transform([text])
    lr_pred = lr_model.predict(vectorized_input)[0]
    rf_pred = rf_model.predict(vectorized_input)[0]

    lr_proba = lr_model.predict_proba(vectorized_input)[0]
    rf_proba = rf_model.predict_proba(vectorized_input)[0]

    return {
        "Logistic Regression": {
            "Prediction": "Fake" if lr_pred == 0 else "Real",
            "Confidence": round(max(lr_proba) * 100, 2)
        },
        "Random Forest": {
            "Prediction": "Fake" if rf_pred == 0 else "Real",
            "Confidence": round(max(rf_proba)*100, 2)
        }
    }

if __name__ == "__main__":
    user_input = input("Enter news content to check: ")
    results = predict_news(user_input)

    print("\nPrediction Results:")
    for model, res in results.items():
        print(f"\n{model}")
        print(f"Prediction: {res['Prediction']}")
        print(f"Confidence: {res['Confidence']}%")
