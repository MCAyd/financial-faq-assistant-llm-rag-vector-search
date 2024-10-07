import os
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
from dotenv import load_dotenv

from db import prep_db, init_db

load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL_LOCAL")
MODEL_NAME = os.getenv("MODEL_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

BASE_URL = "https://github.com/MCAyd/financial-faq-assistant-llm-rag-vector-search/blob/main/"


def fetch_documents():
    print("Fetching documents...")
    relative_url = "documents-with-ids.json"
    docs_url = f"{BASE_URL}/{relative_url}?raw=1"
    docs_response = requests.get(docs_url)
    documents = docs_response.json()
    print(f"Fetched {len(documents)} documents")
    return documents

def load_model():
    print(f"Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def setup_elasticsearch():
    print("Setting up Elasticsearch...")
    es_client = Elasticsearch(ELASTIC_URL)

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "context": {"type": "text"},
                "company_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},

                "question_context_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
            }
        }
    }

    es_client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
    es_client.indices.create(index=INDEX_NAME, body=index_settings)
    print(f"Elasticsearch index '{INDEX_NAME}' created")
    return es_client


def index_documents(es_client, documents, model):
    print("Indexing documents...")
    for doc in tqdm(documents):
        question = doc["question"]
        context = doc["context"]
        doc["question_text_vector"] = model.encode(question + " " + context).tolist()
        es_client.index(index=INDEX_NAME, document=doc)
    print(f"Indexed {len(documents)} documents")


def main():
    print("Starting the indexing process...")

    documents = fetch_documents()
    model = load_model()
    es_client = setup_elasticsearch()
    index_documents(es_client, documents, model)

    print("Initializing database...")
    prep_db()
    print("Connecting database... Creating tables...")
    init_db()

    print("Indexing process completed successfully!")


if __name__ == "__main__":
    main()
