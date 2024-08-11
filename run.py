"""
This module performs document retrieval from some arbitrary queries in the
MS-Marco corpus and reports the time taken.
"""

from datasets import load_dataset
from retriever import Retriever
import time

CORPUS_PATH = "mteb/msmarco-v2"


def load_queries(n=1000):
    # Load 1K queries
    queries_dataset = load_dataset("mteb/msmarco-v2", "queries", streaming=True)
    queries = []

    for example in queries_dataset["queries"]:

        if len(queries) < n:
            queries.append(example["text"])
        else:
            break

    return queries


if __name__ == "__main__":

    model = Retriever()
    model.embed_and_store_corpus(CORPUS_PATH)

    # Get 1K queries
    queries = load_queries()

    call_times = []
    for query in queries:

        # Evaluate time to get response
        start_time = time.time()
        model.search(query=query, k=1)
        end_time = time.time()

        call_times.append(end_time - start_time)

    print(f"mean time: {sum(call_times) / len(call_times):.4f}")
    print(f" max time: {max(call_times):.4f}")
    print(f" min time: {min(call_times):.4f}")
