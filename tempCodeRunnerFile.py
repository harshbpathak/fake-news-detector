f"### 📊 Logistic Regression Model Prediction: **{'Real' if lr_pred == 1 else 'Fake'}**\n"
        f"**Confidence:** {round(lr_proba[lr_pred] * 100, 2)}%\n\n"
        f"### 🌲 Random Forest Model Prediction: **{'Real' if rf_pred == 1 else 'Fake'}**\n"
        f"**Confidence:** {round(rf_proba[rf_pred] * 100, 2)}%\n\n"