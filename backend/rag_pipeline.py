import math
from socket import timeout
from tabnanny import verbose
from typing import List, Dict, Optional
from pydantic import BaseModel
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core import Response
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.base import PromptTemplate
import json

from sympy import true
import config
from llm import RetrievedNode, LLMResponse

    
class RetrievalEvent(Event):
    response: str
    nodes: List[RetrievedNode]
    query: str

    
class RAGWorkflow(Workflow):
    def __init__(self, components: dict, timeout: int = 180, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.components = components
        
    @step()
    async def retrieve_and_generate(self, ctx: Context, ev: StartEvent) -> RetrievalEvent:
        """First step: Perform RAG to get initial response"""
        query = ev.query
        # Set up query engine
        query_engine = self.components["query_engine"]
        
        # Get response
        response: Response = await query_engine.aquery(query)
        
        # Process nodes
        retrieved_nodes = []
        for node in response.source_nodes:
            doi = node.node.metadata.get("DOI", "No DOI")
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
            
        return RetrievalEvent(
            response=str(response),
            nodes=retrieved_nodes,
            query=query
        )
    
    @step()
    async def analyze_design(self, ctx: Context, ev: RetrievalEvent) -> StopEvent:
        """Second step: Analyze feasibility and novelty of the solution"""
        
        # Format prompt for design analysis
        prompt = config.DESIGN_ANALYSIS_PROMPT.format(
            context="\n".join([node.metadata["text"] for node in ev.nodes]),
            query=ev.query,
            response=ev.response
        )
        
        design_prompt = PromptTemplate(prompt)
        
        # Get analysis from LLM
        llm = self.components["llm"]
        analysis_response = await llm.acomplete(prompt)
        
        # try:
        #     design_analysis = json.loads(str(analysis_response))
        # except json.JSONDecodeError:
        #     design_analysis = {
        #         "feasibility_score": 0,
        #         "novelty_score": 0,
        #         "design_approach": "Error parsing LLM response"
        #     }
            
        return StopEvent(
            result=LLMResponse(
                response=ev.response,
                retrieved_nodes=ev.nodes,
                design_approach=str(analysis_response)
            )
        )

async def create_rag_workflow(components: dict) -> RAGWorkflow:
    """Create and configure the RAG workflow"""
    workflow = RAGWorkflow(components=components, timeout=180, verbose=True)
    return workflow