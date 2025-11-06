import re
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer

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
 
# ----------------------------
# Cleaning (matches your rules)
# ----------------------------
URL_RE          = re.compile(r"(https?://\S+|www\.\S+)")
MONEY_RE        = re.compile(r"(?:\$|usd|eur|sgd|£|₹)\s?\d[\d,]*(?:\.\d+)?", re.I)
NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
HTML_RE         = re.compile(r"<[^>]+>")
REPEAT_CHAR_RE  = re.compile(r"(.)\1{3,}")
EMOJI_RE        = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF" "\U00002700-\U000027BF" "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF" "]+",
    flags=re.UNICODE,
)
 
class Cleaner(BaseEstimator, TransformerMixin):
    """Clean a raw string (or list of strings) to match training-time normalization."""
    def __init__(self, strip_html=False, lowercase=True, remove_emojis=True, strip_whitespace=True):
        self.strip_html = strip_html
        self.lowercase = lowercase
        self.remove_emojis = remove_emojis
        self.strip_whitespace = strip_whitespace
 
    def fit(self, X, y=None): return self
 
    def _clean(self, s: str) -> str:
        s = str(s or "")
        s = re.sub(URL_RE, " URL ", s)
        s = re.sub(MONEY_RE, " MONEY ", s)
        s = re.sub(NUMBER_TOKEN_RE, " NUMBER ", s)
        if self.strip_html:
            s = re.sub(HTML_RE, " ", s)
        if self.remove_emojis:
            s = re.sub(EMOJI_RE, " ", s)
        s = re.sub(REPEAT_CHAR_RE, r"\1\1", s)
        if self.lowercase:
            s = s.lower()
        if self.strip_whitespace:
            s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^[\s\u200B-\u200D\uFE0F\uFEFF]+", "", s)
        s = re.sub(r"[\s\u200B-\u200D\uFE0F\uFEFF]+$", "", s)
        return s
 
    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        return [self._clean(x) for x in X]
 
 
# ----------------------------
# MPNet Encoder (lazy-loaded)
# ----------------------------
def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
 
class MPNetEncoder(BaseEstimator, TransformerMixin):
    """SentenceTransformer all-mpnet-base-v2 → 768-dim embeddings (normalized)."""
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=64, normalize=True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = _pick_device()
        self._enc = None
 
    def _ensure(self):
        if self._enc is None:
            self._enc = SentenceTransformer(self.model_name, device=self.device)
 
    def fit(self, X, y=None): return self
 
    def transform(self, X):
        self._ensure()
        embs = self._enc.encode(
            X,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embs, dtype=np.float32)
 
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_enc"] = None  # keep joblib small; model will lazy-load on use
        return st
 
 
# ----------------------------
# XGB + label decoder wrapper
# ----------------------------
class XGBWithDecoder(ClassifierMixin, BaseEstimator):
    """
    Wraps an already-trained xgboost.XGBClassifier (trained on label-encoded ints)
    and a fitted LabelEncoder. predict() returns original string labels;
    predict_proba() returns probabilities in the same order as label_encoder.classes_.
    """
    def __init__(self, clf, label_encoder):
        self.clf = clf
        self.label_encoder = label_encoder
        # cache classes (list[str]) for convenience
        self.classes_ = getattr(self.label_encoder, "classes_", None)
 
    def fit(self, X, y=None):
        # Not used (already trained)
        return self
 
    def predict(self, X):
        enc_pred = self.clf.predict(X)
        # enc_pred is numeric (0..K-1); map back to original labels
        return self.label_encoder.inverse_transform(enc_pred.astype(int))
 
    def predict_proba(self, X):
        # Return proba aligned to label_encoder.classes_
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        raise AttributeError("Underlying classifier does not support predict_proba.")
 
    def decision_function(self, X):
        # Optional: for API completeness if needed by some tooling
        if hasattr(self.clf, "decision_function"):
            return self.clf.decision_function(X)
        raise AttributeError("Underlying classifier does not support decision_function.")