import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google.oauth2 import service_account
from llama_index.vector_stores.google import set_google_config
import llama_index.vector_stores.google.genai_extension as genaix
from typing import Iterable
from random import randrange
from llama_index.indices.managed.google import GoogleIndex
import pandas as pd
from llama_index.core import Response

from retrievers.utils.utils import create_documents

# Update the path to the service account file
service_account_path = os.path.join(os.path.dirname(__file__), '../data/google_ai_service_account.json')

credentials = service_account.Credentials.from_service_account_file(
    service_account_path,
    scopes=[
        "https://www.googleapis.com/auth/generative-language.retriever",
    ],
)
set_google_config(auth_credentials=credentials)

LLAMA_INDEX_COLAB_CORPUS_ID_PREFIX = f"thesis-retriever-test"
SESSION_CORPUS_ID_PREFIX = (
    f"{LLAMA_INDEX_COLAB_CORPUS_ID_PREFIX}-{randrange(1000000)}"
)


def corpus_id(num_id: int) -> str:
    return f"{SESSION_CORPUS_ID_PREFIX}-{num_id}"

def list_corpora() -> Iterable[genaix.Corpus]:
    client = genaix.build_semantic_retriever()
    yield from genaix.list_corpora(client=client)
    
def delete_corpus(*, corpus_id: str) -> None:
    client = genaix.build_semantic_retriever()
    genaix.delete_corpus(corpus_id=corpus_id, client=client)
    
def cleanup_colab_corpora():
    for corpus in list_corpora():
        if corpus.corpus_id.startswith(LLAMA_INDEX_COLAB_CORPUS_ID_PREFIX):
            try:
                delete_corpus(corpus_id=corpus.corpus_id)
                print(f"Deleted corpus {corpus.corpus_id}.")
            except Exception:
                pass

SESSION_CORPUS_ID = corpus_id(1)
cleanup_colab_corpora()

# Create a corpus
index = GoogleIndex.create_corpus(
    corpus_id=SESSION_CORPUS_ID, display_name="THESIS RETREIVER TEST"
)
print(f"Newly created corpus ID is {index.corpus_id}.")

# Ingestion
cwd = os.getcwd()
data_dir= "data/benchmark/benchmark_dataset_1.csv" 
data_path = os.path.join(cwd, data_dir)

df = pd.read_csv(data_path, header=0)
documents = create_documents(df)
index.from_documents(documents,show_progress=True)

# for corpus in list_corpora():
#     print(corpus)
    
query_engine = index.as_query_engine()
response = query_engine.query("adaptive fin-like mechanisms in robotics for enhanced maneuverability")
assert isinstance(response, Response)

# # Show response.
print(f"Response is {response.response}")