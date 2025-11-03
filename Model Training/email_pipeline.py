import re
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

class EmailClassifierPipeline:
    def __init__(self, model_path="./artifacts"):
        """
        Load the BERT model, tokenizer, and label mapping.
        model_path: folder containing config.json + model.safetensors + tokenizer files
        """
        self.model_path = model_path
        self.model = BertForSequenceClassification.from_pretrained(model_path, safe_serialization=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.eval()

        # Hard-coded label map (robust to missing keys)
        self.label_map = {
            0: "forum",
            1: "promotions",
            2: "social_media",
            3: "spam",
            4: "updates",
            5: "verify_code",
            6: "concert_promotion",
            7: "flight_booking"
        }

    def preprocess_text(self, text):
        """Preprocess text exactly as done during training"""
        text = text.lower()
        text = re.sub(r'http[s]?://\S+', 'URL', text)
        text = re.sub(r'www\.\S+', 'URL', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def predict(self, texts):
        """Predict label and probability for one or multiple texts"""
        if isinstance(texts, str):
            texts = [texts]

        texts = [self.preprocess_text(t) for t in texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

        preds = np.argmax(probs, axis=1)

        # Use .get() to avoid KeyError for unknown labels
        readable_labels = [self.label_map.get(int(p), str(p)) for p in preds]

        return readable_labels, probs
