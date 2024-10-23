import asyncio
from collections import defaultdict
import logging
from dotenv import load_dotenv
import os
import sys
import csv

# Load environment variables from .env file
load_dotenv()
# Add the PYTHONPATH from the .env file
sys.path.append(os.getenv("PYTHONPATH"))
from retrievers.utils.utils import get_or_build_index

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from evaluation.utils.utils import save_results

import config

mixedbread_api_key = os.getenv('MXBAI_API_KEY')

TABLE_NAMES = {
    "bm25": "bm25_experiment_MiniLM_reranker_results",
    "vector": "vector_experiment_MiniLM_reranker_results",
    "hybrid": "hybrid_experiment_MiniLM_reranker_results"
}

async def setup_retrievers(index):
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
    vector_retriever = index.as_retriever(similarity_top_k=10)
    
    # postprocessor = SentenceTransformerRerank(
    #     model="mixedbread-ai/mxbai-rerank-large-v1", top_n=5
    # )
    
    postprocessor = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=5
)

    
    retrievers = {
        "bm25": QueryFusionRetriever(
            [bm25_retriever],
            similarity_top_k=5,
            num_queries=5,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=config.QUERY_GEN_PROMPT
        ),
        "vector": QueryFusionRetriever(
            [vector_retriever],
            similarity_top_k=5,
            num_queries=5,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=config.QUERY_GEN_PROMPT
        ),
        "hybrid": QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=5,
            num_queries=5,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=config.QUERY_GEN_PROMPT
        )
    }
    
    return {name: RetrieverQueryEngine.from_args(retriever, node_postprocessors=[postprocessor],response_mode='no_text') for name, retriever in retrievers.items()}

async def process_query(query_str, query_engines):
    results = {}
    for name, query_engine in query_engines.items():
        try:
            response = await query_engine.aquery(query_str)
            retrieved_nodes = response.source_nodes
            retrieved_nodes_titles = [
                {"title": node.metadata.get('Title', 'No title'), "score": node.score}
                for node in retrieved_nodes
            ]
            results[name] = {
                "nodes": retrieved_nodes_titles,
                "queries": response.metadata.get('generated_queries', [])
            }
        except Exception as e:
            logging.error(f"Error processing query '{query_str}' with retriever '{name}': {str(e)}")
            results[name] = {"nodes": ["Error"], "queries": []}
    return results

async def main():

    Settings.llm = Ollama(model="mistral", request_timeout=60.0) # will be used for query generation
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")
    # Settings.embed_model = MixedbreadAIEmbedding(api_key=mixedbread_api_key, model_name="mixedbread-ai/mxbai-embed-large-v1",prompt="Represent this sentence for searching relevant passages:")

    # Read data from the CSV file
    csv_file_path = 'data/benchmark/title_abstracts.csv'
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for index, row in enumerate(csv_reader):
            dataset_name = f"benchmark_dataset_{index+1}"
            title = row['Title']
            queries = [row[f'Query{i}'] for i in range(1, 6)]

            print(f"Processing {dataset_name} with title: {title}")
            
            # Set up the index for each dataset
            try:
                persist_dir = f'data/benchmark/mixedbread/persisted_index_{dataset_name}'
                index = await get_or_build_index(Settings.embed_model, persist_dir=persist_dir, data_dir=f'data/benchmark/{dataset_name}.csv')

                retrievers = await setup_retrievers(index)

                # Run experiments for each query
                for query_str in queries:   
                    results = await process_query(query_str, retrievers)
                    
                    # response, fmt_prompts = await generate_response_cr(retrieved_nodes, query_str, qa_prompt, refine_prompt, Settings.llm) #Create and Refine
                    # await deep_evaluate(query_str,str(response),retrieved_nodes,fusion_retriever....)
                    
                    for name, data in results.items():
                        save_results(query_str, "", data["nodes"], None, data["queries"], TABLE_NAMES[name])
            except Exception as e:
                logging.error(f"Error processing dataset {dataset_name}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
