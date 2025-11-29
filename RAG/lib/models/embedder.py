from typing import List, Union


import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from chromadb import EmbeddingFunction

class Embedder(EmbeddingFunction):
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            embedding_dim = self.model.config.hidden_size
            print(f"{model_name} successfully loaded - embedding dimension: {embedding_dim}")
        
            print(f"{model_name} successfully loaded")
        
        except Exception as e:
            print(f"Something went wrong during {model_name} loading.")
            raise
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with torch.no_grad():
            return self.encode_batch(texts)
    
    def encode_batch(self, texts):
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt" 
        ).to(self.device)

        outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state
        input_mask_exp = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_exp, 1)
        sum_mask = torch.clamp(input_mask_exp.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return embeddings.cpu().numpy().tolist()
    
    def __call__(self, input):
        return self.encode(input)
    
    # def embed_query(self, input: str) -> List[float]:
    #     """For embedding a single query string"""
    #     embeddings = self.encode(input)
    #     return embeddings[0] if embeddings else []
    
