from llama_index.core import (Settings, PromptTemplate)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from evaluation.deep_eval import deep_evaluate
from retrievers.utils.strategy.create_and_refine import generate_response_cr
from retrievers.utils.strategy.hierarchical_summarization import agenerate_response_hs
from retrievers.utils.utils import get_or_build_index
from llama_index.embeddings.openai import OpenAIEmbedding
import config
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

async def query_llm(query_str, generate_queries_flag=True):
    Settings.llm = Ollama(model="mistral", request_timeout=60.0)
    # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    index = await get_or_build_index(embed_model=Settings.embed_model,)

    query_gen_prompt = PromptTemplate(config.QUERY_GEN_PROMPT)
    vector_retriever = index.as_retriever(similarity_top_k=10, verbose=True)
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10, verbose=True)

    postprocessor = SentenceTransformerRerank(
        model="mixedbread-ai/mxbai-rerank-base-v1", top_n=5
    )

    fusion_retriever = QueryFusionRetriever(
        [bm25_retriever, vector_retriever],
        similarity_top_k=5,
        num_queries=6 if generate_queries_flag else 1,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        query_gen_prompt=config.QUERY_GEN_PROMPT
    )

    query_engine = RetrieverQueryEngine.from_args(fusion_retriever, node_postprocessors=[postprocessor])
    
    response = await query_engine.aquery(query_str)
    retrieved_nodes = response.source_nodes
    
    print("Fusing Results done")
    
    print("Generated Queries", response.metadata.get('generated_queries', []))
    print("Running evaluation...")
    
    # Extract metadata and score from retrieved_nodes
    extracted_data = [
        {
            "metadata": node.metadata,
            "score": node.score
        } for node in retrieved_nodes
    ]
        
    data = {
        "response": str(response),
        "retrieved_nodes": extracted_data,
        "generated_queries": response.metadata.get('generated_queries', [])
    }
    return data
