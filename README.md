# Document Retrieval

This repository contains code to search for relevant "documents" from the [MS-Marco Dataset](https://huggingface.co/datasets/mteb/msmarco-v2) given a query. We use a pre-trained [BERT encoder](https://arxiv.org/pdf/1810.04805) to generate embeddings for queries and documents.

### Quick Links

- [Training a model](https://github.com/karangrewal/MSMarco-Document-Retriever/blob/main/train.py#L92-L136) to retrieve the correct document for a given query.
- [Perform document retrieval](https://github.com/karangrewal/MSMarco-Document-Retriever/blob/main/retriever.py#L58-L86) over a large corpus given a query.