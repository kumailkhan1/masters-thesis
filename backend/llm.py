import os
from llama_index.core import (Settings, Response, get_response_synthesizer)
from llama_index.core.indices.utils import default_parse_choice_select_answer_fn
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
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
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.response_synthesizers.type import ResponseMode
import torch
import traceback
import math
from typing import List, Dict
from pydantic import BaseModel
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import llama_index.core

llama_index.core.set_global_handler("simple")


mixedbread_api_key = os.getenv('MXBAI_API_KEY')

class RetrievedNode(BaseModel):
    metadata: Dict[str, str]
    score: float

class LLMResponse(BaseModel):
    response: str
    retrieved_nodes: List[RetrievedNode]
    design_approach: str | None

async def initialize_components():
    """Initialize all components needed for the LLM query system."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        
        print("Setting up LLM...")
        Settings.llm = OpenAI(model="gpt-4o",system_prompt=config.SYSTEM_PROMPT,temperature=0.1)
        
        print("Setting up embedding model...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            query_instruction="Represent this sentence for searching relevant passages:"
        )
        
        print("Getting or building index...")
        index = await get_or_build_index(embed_model=Settings.embed_model)
        
        print("Setting up retrievers...")
        vector_retriever = index.as_retriever(similarity_top_k=20, verbose=True)
        bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=20, verbose=True)
        
        print("Setting up postprocessor...")
        postprocessor = SentenceTransformerRerank(
            model="mixedbread-ai/mxbai-rerank-base-v1", top_n=5
        )
        
        choice_select_prompt = PromptTemplate(
            config.CHOICE_SELECT_PROMPT, prompt_type=PromptType.CHOICE_SELECT)
        
        llmrerank = LLMRerank(top_n=5,choice_select_prompt=choice_select_prompt,choice_batch_size=20)
        
        return {
            "index": index,
            "vector_retriever": vector_retriever,
            "bm25_retriever": bm25_retriever,
            "postprocessor": postprocessor,
            "llmrerank": llmrerank
        }
    except Exception as e:
        print(f"Error initializing components: {e}")
        raise e

async def query_llm(query_str: str, components: dict, generate_queries_flag: bool = True) -> LLMResponse:
    try:
        print("Setting up fusion retriever...")
        fusion_retriever = QueryFusionRetriever(
            [components["bm25_retriever"], components["vector_retriever"]],
            similarity_top_k=20,
            num_queries=6 if generate_queries_flag else 1,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=config.QUERY_GEN_PROMPT
        )
        
        print("Fusion retriever created!!")
        
        synth = get_response_synthesizer(
            response_mode=ResponseMode.ACCUMULATE,
        )
        
        print("Setting up query engine...")
        query_engine = RetrieverQueryEngine.from_args(
            fusion_retriever,
            response_synthesizer=synth,
            node_postprocessors=[components["postprocessor"]],
            use_async=True
        )
        
        print("QE created!!")
        
        print("Querying...")
        response: Response = await query_engine.aquery(query_str)
        
        retrieved_nodes = []
        for node in response.source_nodes:
            doi = node.node.metadata.get("DOI", "No DOI")
            # Check if DOI is nan and replace with a string
            if isinstance(doi, float) and math.isnan(doi):
                doi = "No DOI"
            
            retrieved_nodes.append(
                RetrievedNode(
                    metadata={
                        "Title": node.node.metadata.get("Title", "No title"),
                        "DOI": doi,
                        "Authors": node.node.metadata.get("Authors", "No authors"),
                        "text": node.node.text
                    },
                    score=node.score
                )
            )

        llm_response = LLMResponse(
            response=str(response),
            retrieved_nodes=retrieved_nodes,
            design_approach=None
        )

        # print("Generated Queries", response.metadata.get('generated_queries', []))
        print("Running evaluation...")
        
        save_results(
            query_str, 
            response, 
            response.source_nodes, 
            None,  # metrics_scores
            response.metadata.get('generated_queries', []),  # generated_queries
            "Senckenberg_AccumulateStrat_with_LLMReranker"
        )
        
        return llm_response

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
        return LLMResponse(
            response=f"An error occurred: {str(e)}",
            retrieved_nodes=[]
        )
