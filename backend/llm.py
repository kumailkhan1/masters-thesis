
from llama_index.core import (Settings, PromptTemplate,)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama

from backend.retrievers import FusionRetriever
from backend.retrievers.utils.utils import generate_response_cr, get_or_build_index
import config

async def query_llm(query_str):
    Settings.llm = Ollama(model="mistral", request_timeout=60.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = await get_or_build_index(embed_model=Settings.embed_model)

    query_gen_prompt = PromptTemplate(config.QUERY_GEN_PROMPT)
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

    fusion_retriever = FusionRetriever(
        Settings.llm, query_gen_prompt, [vector_retriever, bm25_retriever], similarity_top_k=5, generate_queries_flag = True)
    retrieved_nodes = await fusion_retriever.aretrieve(query_str)
    
    qa_prompt = PromptTemplate(config.QA_PROMPT)
    refine_prompt = PromptTemplate(config.REFINE_PROMPT)
    response, fmt_prompts = await generate_response_cr(retrieved_nodes, query_str, qa_prompt, refine_prompt, Settings.llm)
    
    print("Generated Queries", fusion_retriever.generated_queries)
    print("Running evaluation...")
    # Store and upload evaluation results
    # await store_and_upload_results(query_str, str(response), retrieved_nodes)
    
    # await deep_evaluate(query_str,str(response),retrieved_nodes,fusion_retriever.generated_queries,"results")
    
    # Extract metadata and score from retrieved_nodes (TODO: Create sep. function)
    extracted_data = []
    for node_with_score in retrieved_nodes:
        metadata = node_with_score.node.metadata  # Assuming metadata is a dictionary
        score = node_with_score.score
        extracted_data.append({
            "metadata": metadata,
            "score": score
        })
        
    data = {
        "response":str(response),
        "retrieved_nodes":retrieved_nodes, #extracted_data,
        "generated_queries":fusion_retriever.generated_queries
    }
    return data
