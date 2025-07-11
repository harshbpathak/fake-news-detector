
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

if __name__ == "__main__":
    print("🔎 AI Misinformation Filter - Type/Paste News Article Below\n")
    user_input = input("📰 News: ")
    label, confidence = predict(user_input)
    print(f"\n🧠 Prediction: {label} (Confidence: {confidence:.2%})")
