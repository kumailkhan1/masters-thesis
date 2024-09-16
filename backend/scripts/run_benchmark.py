from dotenv import load_dotenv
import os
import sys
import csv

# Load environment variables from .env file
load_dotenv()

# Add the PYTHONPATH from the .env file
sys.path.append(os.getenv("PYTHONPATH"))

import asyncio
from retrievers.utils.strategy.create_and_refine import generate_response_cr
from llm import query_llm
from evaluation.deep_eval import deep_evaluate
from retrievers.utils.utils import get_or_build_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from retrievers.FusionRetriever import FusionRetriever
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings,PromptTemplate
from evaluation.utils.utils import save_results
import config

TABLE_NAME_BM25 = "bm25_experiment_reranking_queries_results"
TABLE_NAME_VECTOR = "vector_experiment_reranking_queries_results"
TABLE_NAME_HYBRID = "hybrid_experiment_reranking_queries_results"


async def main():
    Settings.llm = Ollama(model="mistral", request_timeout=60.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Read data from the CSV file
    csv_file_path = 'data/benchmark/title_abstracts.csv'
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for index, row in enumerate(csv_reader):
            print(row.keys())  # Add this line
            dataset_name = f"benchmark_dataset_{index+1}"
            title = row['Title']
            queries = [row[f'Query{i}'] for i in range(1, 6)]

            print(f"Processing {dataset_name} with title: {title}")
            
            # Set up the index for each dataset
            persist_dir = f'data/benchmark/persisted_index_{dataset_name}'
            index = await get_or_build_index(Settings.embed_model, persist_dir=persist_dir, data_dir=f'data/benchmark/{dataset_name}.csv')

            # Prepare the retrievers
            bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
            vector_retriever = index.as_retriever(similarity_top_k=10)
            fusion_retriever = FusionRetriever(Settings.llm, config.QUERY_GEN_PROMPT, [vector_retriever,bm25_retriever], similarity_top_k=5, generate_queries_flag=True)
   
            # Run experiments for each query
            for query_str in queries:   
                retrieved_nodes = await fusion_retriever.aretrieve(query_str)
                retrieved_nodes_titles = [
                    {"title": node.node.metadata.get('Title', 'No title'), "score": node.score}
                    for node in retrieved_nodes
                ]
                # response, fmt_prompts = await generate_response_cr(retrieved_nodes, query_str, qa_prompt, refine_prompt, Settings.llm) #Create and Refine
                # await deep_evaluate(query_str,str(response),retrieved_nodes,fusion_retriever.generated_queries,TABLE_NAME_HYBRID) #DeepEval
                save_results(query_str,"",retrieved_nodes_titles,None,fusion_retriever.generated_queries,TABLE_NAME_HYBRID)

# Run the experiment
asyncio.run(main())
