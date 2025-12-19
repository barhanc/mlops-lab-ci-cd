import os
import numpy as np
import onnxruntime as ort

from cleantext import clean
from tokenizers import Tokenizer

from .settings import Settings


class Inference:
    def __init__(self, settings: Settings) -> None:
        self.tokenizer = Tokenizer.from_file(os.path.join(settings.tokenizer_path, "tokenizer.json"))
        self.embedding_session = ort.InferenceSession(settings.onnx_embedding_model_path)
        self.classifier_session = ort.InferenceSession(settings.onnx_classifier_path)

    def predict(self, text: str) -> str:
        # Clean input
        cleaned_text = clean(text)

        # Tokenize input
        encoded = self.tokenizer.encode(cleaned_text)

        # Prepare numpy arrays for ONNX
        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])

        # Run embedding inference
        embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        embeddings = self.embedding_session.run(None, embedding_inputs)[0]

        # Run classifier inference
        classifier_input_name = self.classifier_session.get_inputs()[0].name
        classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}  # type:ignore
        prediction = self.classifier_session.run(None, classifier_inputs)[0]

        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        label = sentiment_map.get(prediction[0], "unknown")  # type:ignore

        return label
