import os
from llama_index.core import (Settings, PromptTemplate, Response, get_response_synthesizer)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from evaluation.utils.utils import save_results
from evaluation.deep_eval import deep_evaluate
from retrievers.utils.strategy.create_and_refine import generate_response_cr
from retrievers.utils.strategy.hierarchical_summarization import agenerate_response_hs
from retrievers.utils.utils import get_or_build_index
from llama_index.embeddings.openai import OpenAIEmbedding
import config
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
import torch
import traceback
from typing import List, Dict, Any
from pydantic import BaseModel

mixedbread_api_key = os.getenv('MXBAI_API_KEY')

class RetrievedNode(BaseModel):
    metadata: Dict[str, str]
    score: float

class LLMResponse(BaseModel):
    response: str
    retrieved_nodes: List[RetrievedNode]

async def query_llm(query_str: str, generate_queries_flag: bool = True) -> LLMResponse:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        
        print("Setting up LLM...")
        Settings.llm = Ollama(model="mistral", request_timeout=60.0)
        
        print("Setting up embedding model...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1",
                                                    query_instruction="Represent this sentence for searching relevant passages:")
        
        print("Getting or building index...")
        index = await get_or_build_index(embed_model=Settings.embed_model)
        
        print("Index Loaded!!")

        print("Setting up retrievers...")
        vector_retriever = index.as_retriever(similarity_top_k=20, verbose=True)
        bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=20, verbose=True)
    
        print("Retrievers loaded!!")
        
        print("Setting up postprocessor...")
        postprocessor = SentenceTransformerRerank(
            model="mixedbread-ai/mxbai-rerank-base-v1", top_n=5
        )

        print("Setting up fusion retriever...")
        fusion_retriever = QueryFusionRetriever(
            [bm25_retriever, vector_retriever],
            similarity_top_k=20,
            num_queries=6 if generate_queries_flag else 1,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=config.QUERY_GEN_PROMPT
        )
        
        print("Fusion retriever created!!")
        
        qa_prompt = PromptTemplate(config.QA_PROMPT,template_var_mappings={"context_str":"context_str","query_str":"query_str"})
        refine_prompt = PromptTemplate(config.REFINE_PROMPT,template_var_mappings={"context_str":"context_str","query_str":"query_str","existing_answer":"existing_answer"})
        
        synth = get_response_synthesizer(
        text_qa_template=qa_prompt, refine_template=refine_prompt
)
        print("Setting up query engine...")
        query_engine = RetrieverQueryEngine.from_args(
            fusion_retriever,
            response_synthesizer=synth,
            response_mode='refine', 
            node_postprocessors=[postprocessor],
            use_async=True
        )
        
        print("QE created!!")
        
        print("Querying...")
        response: Response = await query_engine.aquery(query_str)
        
        retrieved_nodes = [
            RetrievedNode(
                metadata={
                    "Title": node.node.metadata.get("Title", "No title"),
                    "DOI": node.node.metadata.get("DOI", "No DOI"),
                    "Authors": node.node.metadata.get("Authors", "No authors"),
                    "Abstract": node.node.text
                },
                score=node.score
            )
            for node in response.source_nodes
        ]

        llm_response = LLMResponse(
            response=str(response),
            retrieved_nodes=retrieved_nodes
        )

        save_results(query_str, response, response.source_nodes, None, None, "Senckenberg_Feedback_Queries")
        return llm_response

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
        return LLMResponse(
            response=f"An error occurred: {str(e)}",
            retrieved_nodes=[]
        )
