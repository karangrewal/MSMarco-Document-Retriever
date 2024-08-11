"""
This module trains a MLP to predict the correct document embedding given a query embedding
on the corpus & query embeddings in order to retrieve the correct one
"""

from datasets import load_dataset
import json
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


CORPUS_PATH = "mteb/msmarco-v2"

BATCH_SIZE = 64
LR = 3e-4
NUM_ITERS = 1000


class QueryHead(nn.Module):
    """ 2-Layer MLP to transform query embeddings """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, X):
        """
        X -> (B, D)
        """
        return self.layers(X)


def get_training_batch(dataset, query_embeddings, document_embeddings, batch_size):
    """
    Returns a tuple of tensors (query_embeddings, document_embeddings) where
    query_embedding[i] and document_embedding[i] are a true retrieval pair.
    """
    list_of_query_ids = json.load(open("/workspace/train_data/list_of_query_ids.json", "r"))
    list_of_document_ids = json.load(open("/workspace/train_data/list_of_document_ids.json", "r"))

    query_document_pairs = []
    while len(query_document_pairs) < batch_size:

        idx = random.randint(0, len(dataset) - 1)
        
        query_id = dataset[idx]["query-id"]
        document_id = dataset[idx]["corpus-id"]

        if query_id in list_of_query_ids and document_id in list_of_document_ids:
            query_document_pairs.append((query_id, document_id))

    # Now get indices of each query-document pair to find the corresponding embeddings
    inds = [(list_of_query_ids.index(query_id), list_of_document_ids.index(document_id)) for (query_id, document_id) in query_document_pairs]

    # Index embeddings table to get embeddings
    embeddings = [(query_embeddings[qidx, :], document_embeddings[didx, :]) for (qidx, didx) in inds]

    query_embeddings = [qe for (qe, _) in embeddings]
    query_embeddings = torch.vstack(query_embeddings)

    document_embeddings = [de for (_, de) in embeddings]
    document_embeddings = torch.vstack(document_embeddings)

    return query_embeddings, document_embeddings


def train(model, optimizer, device, num_iters, batch_size):
    dataset = load_dataset(CORPUS_PATH, "default")["train"]
    query_embeddings = torch.load("/workspace/train_data/queries.pt")
    document_embeddings = torch.load("/workspace/train_data/documents.pt")

    print(f"training on {device} ...")

    for iter_id in range(1, num_iters + 1):

        optimizer.zero_grad()

        qe, de = get_training_batch(
            dataset=dataset,
            query_embeddings=query_embeddings,
            document_embeddings=document_embeddings,
            batch_size=batch_size
        )

        qe, de = qe.to(device), de.to(device)
        labels = torch.arange(batch_size, device=device)

        # Pass query embeddings through model
        qe_transformed = model(qe)

        # Make predictions - treat query-document pairs as positive examples and
        # everything else as negative examples; every query-document combination
        # from this batch can be given a true/false label, and this becomes a
        # classification task which train via cross-entropy loss
        preds = qe_transformed @ de.T # (B, B) where the diagonal elements should be
                                      # close to 1, non-diagonal elements should be
                                      # close to zero

        loss = F.cross_entropy(input=preds, target=labels)
        loss.backward()
        optimizer.step()

        if iter_id % 10 == 0:
            print(f" iter {iter_id}: loss={loss.item():.4f}")

            # Save checkpoint
            torch.save(model.state_dict(), "query_head.pth")


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QueryHead(input_size=768, hidden_size=768, output_size=768)
    model.to(device)

    optimizer = Adam(params=model.parameters(), lr=LR)

    train(model=model, optimizer=optimizer, device=device, num_iters=NUM_ITERS,
          batch_size=BATCH_SIZE)
