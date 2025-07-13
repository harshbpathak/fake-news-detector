import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from llm_layer import fact_check_with_llm
import joblib

# Load the model and tokenizer
MODEL_PATH = "./misinformation_model_final"
LABELS = ["Fake", "Real"]
# Load logistic regression and random forest models
lr_model = joblib.load('./models/logistic_model.pkl')
rf_model = joblib.load('./models/random_forest_model.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def classify_news(text):
    # Step 1: Get ML model prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()
    # results of LR and RF models
    vectorized_input = vectorizer.transform([text])
    lr_pred = lr_model.predict(vectorized_input)[0]
    rf_pred = rf_model.predict(vectorized_input)[0]

    lr_proba = lr_model.predict_proba(vectorized_input)[0]
    rf_proba = rf_model.predict_proba(vectorized_input)[0]

    llm_explanation = fact_check_with_llm(text)

    result = (
        f"### 🤖 Transformer model Prediction: **{LABELS[pred].upper()}**\n"
        f"**Confidence:** {round(confidence * 100, 2)}%\n\n"
        f"### 📊 Logistic Regression Model Prediction: **{'Real' if lr_pred == 1 else 'Fake'}**\n"
        f"**Confidence:** {round(lr_proba[lr_pred] * 100, 2)}%\n\n"
        f"### 🌲 Random Forest Model Prediction: **{'Real' if rf_pred == 1 else 'Fake'}**\n"
        f"**Confidence:** {round(rf_proba[rf_pred] * 100, 2)}%\n\n"
        f"---\n\n"
        f"### 🧠 LLM Check:\n{llm_explanation}"
    )

    return result
custom_css = """
body { background: #181c24; }
.gradio-container { background: #181c24; }
#component-0, .input-textbox textarea {
    font-family: 'Lexend', Arial, sans-serif;
    font-size: 1.1em;
    background: #23283a;
    border-radius: 8px;
    border: 1.5px solid #e67e22;
    color: #f5f6fa;
}
#component-2, .output-markdown {
    font-family: 'Lexend', Arial, sans-serif;
    font-size: 1.15em;
    background: #23283a;
    border-radius: 8px;
    border: 1.5px solid #e67e22;
    color: #f5f6fa;
    padding: 18px;
}
.gr-button {
    background: linear-gradient(90deg, #e67e22 60%, #283566 100%);
    color: #fff;
    font-weight: 700;
    border-radius: 8px;
    border: none;
    font-size: 1.1em;
    padding: 12px 28px;
    transition: background 0.2s;
}
.gr-button:hover {
    background: linear-gradient(90deg, #283566 60%, #e67e22 100%);
    color: #fff;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as iface:
    gr.Markdown(
        """
        # 🧠 <span style="color:#e67e22">AI Misinformation Detector</span>
        <span style="color:#f5f6fa">
        Welcome to AI-powered news checker!  
        Paste any news article or claim below and this model will classify it as **FAKE** or **REAL**.<br>
        </span>
        """,
        elem_id="main-title"
    )
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(
                lines=7,
                label="Paste a news article or headline here...",
                placeholder="e.g. Scientists discover a new planet made entirely of chocolate.",
                elem_id="input-textbox"
            )
            submit_btn = gr.Button("Analyze", elem_id="analyze-btn")
        with gr.Column():
            output_box = gr.Markdown(label="Result", elem_id="output-markdown")

    submit_btn.click(classify_news, inputs=input_box, outputs=output_box)
    input_box.submit(classify_news, inputs=input_box, outputs=output_box)

# Launch locally
if __name__ == "__main__":
    iface.launch()
