"""
Given a set of queries and documents (given by the dicts `id_to_queries` and
`id_to_documents` which are saved to disk), generate embeddings which will be
further used for search.
"""

from datasets import load_dataset
import json
import os
import random
from transformers import AutoTokenizer, AutoModel
import torch


CORPUS_PATH = "mteb/msmarco-v2"
MODEL_NAME = "bert-base-uncased"


def embed_queries(encoder, tokenizer, id_to_queries, batch_size=512,
                  max_items=None):

    # Split `id_to_queries` into 2 parallel lists of IDs and their corresponding
    # queries
    id_query_pairs = list(id_to_queries.items())

    # Cut off at `max_items`
    if max_items is not None:
        random.shuffle(id_query_pairs)
        id_query_pairs = id_query_pairs[:max_items]

    ids = [_id for (_id, _) in id_query_pairs]
    queries = [query for (_, query) in id_query_pairs]
    queries = ["[QUERY] " + q for q in queries] # Add prefix

    # Generate embeddings for each query batch-by-batch
    embeddings = torch.rand(size=(0, encoder.config.hidden_size))

    for i in range(0, len(queries), batch_size):
        queries_i = queries[i:i + batch_size]
        embeddings = torch.cat(
            (embeddings, generate_embeddings(queries_i, encoder, tokenizer))
        )
        print("queries", embeddings.size(), embeddings.device)

    # Save IDs and embeddings; they are parallel
    filename = os.path.join("/workspace/train_data/", "list_of_query_ids.json")
    with open(filename, "w+") as f:
        json.dump(ids, f)

    torch.save(embeddings, os.path.join("/workspace/train_data/", "queries.pt"))


def embed_documents(encoder, tokenizer, id_to_documents, batch_size=512,
                    max_items=None):

    # Split `id_to_documents` into 2 parallel lists of IDs and their
    # corresponding documents
    id_document_pairs = list(id_to_documents.items())

    # Cut off at `max_items`
    if max_items is not None:
        random.shuffle(id_document_pairs)
        id_document_pairs = id_document_pairs[:max_items]

    ids = [_id for (_id, _) in id_document_pairs]
    documents = [document for (_, document) in id_document_pairs]
    documents = ["[DOCUMENT] " + d for d in documents] # Add prefix

    # Cut off at `max_items`
    if max_items is not None:
        documents = documents[:max_items]

    # Generate embeddings for each document batch-by-batch
    embeddings = torch.rand(size=(0, encoder.config.hidden_size))

    for i in range(0, len(documents), batch_size):
        documents_i = documents[i:i + batch_size]
        embeddings = torch.cat(
            (embeddings, generate_embeddings(documents_i, encoder, tokenizer))
        )
        print("documents", embeddings.size(), embeddings.device)

    # Save IDs and embeddings; they are parallel
    filename = os.path.join("/workspace/train_data/", "list_of_document_ids.json")
    with open(filename, "w+") as f:
        json.dump(ids, f)

    torch.save(embeddings, os.path.join("/workspace/train_data/", "documents.pt"))


def generate_embeddings(list_of_inputs, encoder, tokenizer):
    with torch.no_grad():
        tokens = tokenizer(list_of_inputs, padding=True, truncation=True,
                           return_tensors="pt")
        tokens = {k: t.to(encoder.device) for k, t in tokens.items()}

        embeddings = encoder(**tokens).last_hidden_state[:, 0, :] # (B, D), use CLS token
        embeddings_cpu = embeddings.cpu()
        del embeddings
        torch.cuda.empty_cache()
        return embeddings_cpu


if __name__ == "__main__":

    encoder = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)

    # Generate query embeddings
    id_to_queries = json.load(
        open(os.path.join("/workspace/train_data/", "id_to_queries.json"), "r")
    )
    embed_queries(encoder, tokenizer, id_to_queries, max_items=100000) 

    # Generate document embeddings
    id_to_documents = json.load(
        open(os.path.join("/workspace/train_data/", "id_to_documents.json"), "r")
    )
    embed_documents(encoder, tokenizer, id_to_documents, max_items=100000)
