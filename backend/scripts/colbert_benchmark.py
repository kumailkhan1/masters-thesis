from dotenv import load_dotenv
import os
import sys
import csv

# Load environment variables from .env file
load_dotenv()

# Add the PYTHONPATH from the .env file
sys.path.append(os.getenv("PYTHONPATH"))

import pandas as pd
import pandas as pd
from llama_index.indices.managed.colbert.base import ColbertIndex
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from evaluation.utils.utils import save_results
from retrievers.utils.utils import create_documents
from multiprocessing import freeze_support

import config

TABLE_NAME_COLBERT = "colbert_experiment_results"

def get_colbert_index(persist_dir, data_dir):
    df = pd.read_csv(data_dir)
    
    documents = create_documents(df)
    
    index = ColbertIndex.from_documents(
        documents,
        show_progress=True
    )
    index.storage_context.persist(persist_dir=persist_dir)
    return index

def main():
    Settings.llm = Ollama(model="mistral", request_timeout=60.0)
    
    # Read data from the CSV file
    csv_file_path = 'data/benchmark/title_abstracts.csv'
    df = pd.read_csv(csv_file_path)
    
    for index, row in df.iterrows():
        dataset_name = f"benchmark_dataset_{index+1}"
        title = row['Title']
        queries = [row[f'Query{i}'] for i in range(1, 6)]

        print(f"Processing {dataset_name} with title: {title}")
        
        # Set up the index for each dataset
        persist_dir = f'data/benchmark/colbert_index_{dataset_name}'
        index = get_colbert_index(persist_dir=persist_dir, data_dir=f'data/benchmark/{dataset_name}.csv')
        
        if index is None:
            print(f"Failed to create index for {dataset_name}. Skipping...")
            continue
        
        # Run experiments for each query
        for query_str in queries:   
            retrieved_nodes = index.as_retriever().retrieve(query_str, top_k=5)
            retrieved_nodes_titles = [
                {
                    "title": node.metadata.get("Title", "No title"),
                    "score": node.score,
                }
                for node in retrieved_nodes
            ]
            # Save results
            save_results(query_str, "", retrieved_nodes_titles, None, [], TABLE_NAME_COLBERT)

if __name__ == "__main__":
    freeze_support()
    main()
