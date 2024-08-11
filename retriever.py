"""
This module contains the classes and methods that perform document retrieval.
"""

from dataclasses import dataclass
import json
from transformers import AutoTokenizer, AutoModel
import torch

from train import QueryHead
 

CORPUS_PATH = "mteb/msmarco-v2"
MODEL_NAME = "bert-base-uncased"


@dataclass
class Result:
    document_id: str
    document_text: str
    score: float


class Retriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load encoder 
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.encoder.to(self.device)

        # Load query head from checkpoint
        self.qh = QueryHead(input_size=768, hidden_size=768, output_size=768)
        self.qh.load_state_dict(torch.load("/workspace/query_head.pth"))
        self.qh.to(self.device)

        # Initialize the following to `None`, and they can later be loaded by
        # calling method `embed_and_store_corpus`
        self.id_to_documents = None
        self.list_of_document_ids = None
        self.document_embeddings = None

    def embed_and_store_corpus(self, corpus_path):
        """
        corpus_path -> str representing path to a Hugging Face dataset
        """
        # Dict mapping document IDs to strs
        self.id_to_documents = json.load(open("/workspace/train_data/id_to_documents.json", "r"))

        # The following are parallel:
        # 1. List of document IDs
        # 2. Document embeddings
        self.list_of_document_ids = json.load(open("/workspace/train_data/list_of_document_ids.json", "r"))
        self.document_embeddings = torch.load("/workspace/train_data/documents.pt") # (*, D)
        self.document_embeddings = self.document_embeddings.to(self.device)

    def search(self, query: str, k: int):
        # Encode query + apply transformation head
        tokens = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        tokens = {k: t.to(self.device) for k, t in tokens.items()}
        query_embedding = self.encoder(**tokens).last_hidden_state[:, 0, :] # Use CLS token
        qe_transformed = self.qh(query_embedding) # (1, D)

        # Compare transformed query embedding to all document embeddings
        scores = self.document_embeddings @ qe_transformed.T # (*, 1)
        scores = scores.squeeze() # (*,)

        # Return top k documents in descending order of score
        inds = scores.argsort(descending=True)[:k]

        top_k_ids = [self.list_of_document_ids[idx.item()] for idx in inds]
        top_k_texts = [self.id_to_documents[_id] for _id in top_k_ids]
        top_k_scores = scores.tolist()
        top_k_scores = [top_k_scores[idx.item()] for idx in inds]

        results = [Result(_id, text, score) for _id, text, score in zip(top_k_ids, top_k_texts, top_k_scores)]
        return results


if __name__ == "__main__":

    # Initialize retriever
    model = Retriever()
    model.embed_and_store_corpus(CORPUS_PATH)
    
    # Perform search
    print(model.search(query="Where was Barack Obama born?", k=3))
    print(model.search(query="Which day comes before Tuesday?", k=3))
