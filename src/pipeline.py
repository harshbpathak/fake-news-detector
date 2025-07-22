from transformers import pipeline
import torch
def pipeline_load():
    # Loading model and tokenizer from local directory
    MODEL_PATH = "./misinformation_model_final"
    LABELS = ["Fake", "Real"]
    classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, return_all_scores=True)
    return classifier, LABELS
def classify_news(text, classifier, labels):
    """Classifies news text using the loaded model and returns predictions with confidence scores.
    """
    inputs = classifier(text)
    predictions = []
    for item in inputs:
        label = labels[item['label']]
        confidence = item['score']
        predictions.append(f"{label} ({confidence:.2%})")
    return predictions
def predict(text: str, classifier, labels) -> str:
    """Predicts whether a news article is Fake or Real and returns confidence."""
    inputs = classifier(text)
    predicted_class = max(inputs, key=lambda x: x['score'])
    label = labels[predicted_class['label']]
    confidence = predicted_class['score']
    return label, confidence    
def evaluate_tsv(tsv_path: str, classifier, labels):
    """predicts labels for a TSV file and prints the accuracy."""
    import csv

    total = 0
    correct = 0
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["text"]
            true_label = row["label"]
            pred_label, _ = predict(text, classifier, labels)
            if pred_label == true_label:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
if __name__ == "__main__":
    print("🔎 AI Misinformation Filter")
    print("1. Test a single news article")
    print("2. Evaluate on a TSV file")
    choice = input("Choose an option (1/2): ").strip()
    
    classifier, labels = pipeline_load()
    
    if choice == "1":
        user_input = input("📰 News: ")
        label, confidence = predict(user_input, classifier, labels)
        print(f"\n🧠 Prediction: {label} (Confidence: {confidence:.2%})")
    elif choice == "2":
        tsv_path = input("Enter path to TSV file: ").strip()
        evaluate_tsv(tsv_path, classifier, labels)
    else:
        print("Invalid choice.")    
