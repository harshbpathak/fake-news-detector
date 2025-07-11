# evaluate.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Load model and tokenizer

model_path = "./misinformation_model_final" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)



# Load and prepare test data

df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true])
df = df.sample(frac=1).reset_index(drop=True)

_, test_texts, _, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)


# Predict

preds, labels = [], []

for text, label in zip(test_texts, test_labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    preds.append(prediction)
    labels.append(label)
# Evaluate
print("\nClassification Report:\n")
print(classification_report(labels, preds, target_names=["Real", "Fake"]))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
