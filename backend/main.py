import os
import sys
from fastapi import FastAPI, HTTPException
from llama_index.core import (Settings, get_response_synthesizer)
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.utils.workflow import draw_all_possible_flows
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager


# Load environment variables from .env file
load_dotenv()

# Add the PYTHONPATH from the .env file
sys.path.append(os.getenv("PYTHONPATH"))
import config
from llm import query_llm, LLMResponse, initialize_components
from rag_pipeline import create_rag_workflow


# Global variables to store components
global_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing components...")
    global_components.update(await initialize_components())
    
    # Create workflow-specific components
    fusion_retriever = QueryFusionRetriever(
        [global_components["bm25_retriever"], global_components["vector_retriever"]],
        similarity_top_k=20,
        num_queries=6,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        query_gen_prompt=config.QUERY_GEN_PROMPT
    )
    
    qa_template = PromptTemplate(config.QA_PROMPT,PromptType.QUESTION_ANSWER)
    synth = get_response_synthesizer(text_qa_template=qa_template,response_mode=ResponseMode.ACCUMULATE)
    
    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        response_synthesizer=synth,
        node_postprocessors=[global_components["llmrerank"]],
        use_async=True
    )
    
    global_components.update({
        "query_engine": query_engine,
        "llm": Settings.llm
    })
    
    print("Components initialized successfully!")
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)


# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query", response_model=LLMResponse)
async def handle_query(request: QueryRequest):
    query = request.query
    try:
        workflow = await create_rag_workflow(global_components)
        response = await workflow.run(query=query)
        # draw_all_possible_flows(workflow,filename='recentworkflow.html')
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
