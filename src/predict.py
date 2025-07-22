from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Loading model and tokenizer from local directory
MODEL_PATH = "./misinformation_model_final"
LABELS = ["Fake", "Real"]  

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  #

def predict(text: str) -> str:
    """Predicts whether a news article is Fake or Real and returns confidence."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze()
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    return LABELS[predicted_class], confidence

def evaluate_tsv(tsv_path: str):
    """
    Evaluates the model on a TSV file and prints the accuracy.
    The TSV file should have columns: 'text' and 'label'.
    """
    import csv

    total = 0
    correct = 0
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = row["text"]
            true_label = row["label"]
            pred_label, _ = predict(text)
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
    if choice == "1":
        user_input = input("📰 News: ")
        label, confidence = predict(user_input)
        print(f"\n🧠 Prediction: {label} (Confidence: {confidence:.2%})")
    elif choice == "2":
        tsv_path = input("Enter path to TSV file: ").strip()
        evaluate_tsv(tsv_path)
    else:
        print("Invalid choice.")
