"""
This module loads the MS Marco corpus, divides it into chunks of 100K examples
each, and saves those files.
"""

from datasets import load_dataset
import json
import os

CORPUS_PATH = "mteb/msmarco-v2"
MODEL_NAME = "bert-base-uncased"


def preprocess(chunk_size, out_path):

    # Load corpus in streaming mode
    corpus = load_dataset("mteb/msmarco-v2", "corpus", streaming=True)

    chunk = []
    file_count = 1

    # Iterate through the dataset and save in chunks
    for example in corpus["corpus"]:
        chunk.append(example)

        if len(chunk) == chunk_size:
            # Save the chunk to a JSON file
            filename = f"msmarco_v2_chunk_{file_count}.json"
            with open(filename, "w+") as f:
                json.dump(chunk, f)
            
            print(f"saved {filename}")

            chunk = []
            file_count += 1


def collect_training_ids():

    # 1. Load dataset of query-document associations
    dataset = load_dataset(CORPUS_PATH, "default")["train"]

    # Get IDs of all relevant queries and documents
    query_ids = set()
    corpus_ids = set()
    for example in dataset:
        query_ids.add(example["query-id"])
        corpus_ids.add(example["corpus-id"])

    # 2. Load queries; save ones which are relevant for training
    queries = load_dataset(CORPUS_PATH, "queries")
    id_to_query = {}
    for query in queries["queries"]:
        if query["_id"] in query_ids:
            id_to_query[query["_id"]] = query["text"]

    save_dict(id_to_query, os.path.join("/workspace/train_data/", "id_to_queries.json"))

    # 3. Load documents; save ones which are relevant for training
    print("loading corpus ...")
    documents = load_dataset(CORPUS_PATH, "corpus", streaming=False)
    print("corpus loaded!")

    id_to_documents = {}
    for doc in documents["corpus"]:
        if doc["_id"] in corpus_ids:
            id_to_documents[doc["_id"]] = doc["text"]

    save_dict(id_to_documents, os.path.join("/workspace/train_data/", "id_to_documents.json"))


def save_dict(data, out_path):
    with open(out_path, "w+") as f:
        json.dump(data, f)


if __name__ == "__main__":
    collect_training_ids()
