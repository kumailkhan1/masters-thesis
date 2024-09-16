import os
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from typing import List, Dict
from retrievers.utils.utils import generate_queries
from llama_index.core.postprocessor import SentenceTransformerRerank

from dotenv import load_dotenv
load_dotenv()
import asyncio


postprocessor = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=5
)

class FusionRetriever(BaseRetriever):
    def __init__(
        self,
        llm,
        query_gen_prompt,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 5,
        generate_queries_flag = True,
        rerank_top_n: int = 5  # New parameter for reranking
    ) -> None:
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        self.query_gen_prompt = query_gen_prompt
        self.generate_queries_flag = generate_queries_flag
        self.generated_queries = []
        self.reranker = LLMRerank(
            choice_batch_size=5,
            top_n=rerank_top_n,
            llm=llm  # Use the same LLM or a different one
        )
        super().__init__()
        
    def normalize_scores(self, nodes_with_scores: List[NodeWithScore]) -> List[NodeWithScore]:
        if not nodes_with_scores:
            return []
        
        min_score = min(node.score or 0 for node in nodes_with_scores)
        max_score = max(node.score or 0 for node in nodes_with_scores)
        
        if max_score == min_score:
            return nodes_with_scores
        
        for node in nodes_with_scores:
            if node.score is not None:
                node.score = (node.score - min_score) / (max_score - min_score)
        
        return nodes_with_scores

    def fuse_results(self, results_dict: Dict[str, List[NodeWithScore]], similarity_top_k: int) -> List[NodeWithScore]:
        print("Fusing results...")
        k = 60.0
        fused_scores: Dict[str, float] = {}
        text_to_node: Dict[str, NodeWithScore] = {}

        try:
            for query, nodes_with_scores in results_dict.items():
                normalized_nodes = self.normalize_scores(nodes_with_scores)
                for rank, node_with_score in enumerate(sorted(normalized_nodes, key=lambda x: x.score or 0.0, reverse=True)):
                    text = node_with_score.node.get_content()
                    text_to_node[text] = node_with_score
                    if text not in fused_scores:
                        fused_scores[text] = 0.0
                    fused_scores[text] += 1.0 / (rank + k)

            reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

            reranked_nodes: List[NodeWithScore] = []
            for text, score in reranked_results[:similarity_top_k]:
                node = text_to_node[text]
                node.score = score
                reranked_nodes.append(node)

            return reranked_nodes
        except Exception as e:
            print(f"Error occurred while Fusing Results: {e}")
            return []

    async def run_queries(self, queries: List[str], retrievers: List[BaseRetriever]) -> Dict[str, List[NodeWithScore]]:
        print("Running Queries...")
        results_dict: Dict[str, List[NodeWithScore]] = {}
        for query in queries:
            query_results: List[NodeWithScore] = []
            for retriever in retrievers:
                retrieved_nodes = await retriever.aretrieve(query)
                query_results.extend(retrieved_nodes)
            results_dict[query] = query_results
        return results_dict

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        print("Inside _retrieve")
        if(not self.generate_queries_flag):
            self.generated_queries = generate_queries(self.query_gen_prompt, self._llm, query_bundle.query_str, num_queries=6)
            print(self.generated_queries)
        results =  asyncio.run(self.run_queries(self.generated_queries, self._retrievers))
        final_results = self.fuse_results(results, self._similarity_top_k)
        return final_results
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        print("Running Retrieval _aretrieve()...")
        print("Query Generation: ", self.generate_queries_flag)
        
        if self.generate_queries_flag:
            # Generate additional queries with the help of llm
            queries = generate_queries(self.query_gen_prompt, self._llm, query_bundle.query_str, num_queries=5)
            self.generated_queries = queries  # Store generated queries
        else:
            queries = [query_bundle.query_str]
            self.generated_queries = queries  # Store the main query
        
        results = await self.run_queries(self.generated_queries, self._retrievers)
        final_results = self.fuse_results(results, similarity_top_k=self._similarity_top_k)
        
        # Apply reranking
        try:
            reranked_results = postprocessor.postprocess_nodes(final_results, query_bundle)
            return reranked_results
        except ValueError as e:
            print(f"Error during reranking: {e}")
            # If reranking fails, return the fused results without reranking
            return final_results
