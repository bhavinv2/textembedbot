import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbeddingModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("textembedding-gecko@003")
        self.model = AutoModel.from_pretrained("textembedding-gecko@003")

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
