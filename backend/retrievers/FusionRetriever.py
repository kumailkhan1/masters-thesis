from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from typing import List
from retrievers.utils.utils import generate_queries

import asyncio

class FusionRetriever(BaseRetriever):
    def __init__(
        self,
        llm,
        query_gen_prompt,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
        generate_queries_flag = True
    ) -> None:
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        self.query_gen_prompt = query_gen_prompt
        self.generate_queries_flag = generate_queries_flag
        self.generated_queries = []
        super().__init__()
        
    def fuse_results(results_dict, similarity_top_k):
        print("Fusing results...")
        """
        Apply reciprocal rank fusion.

        The original paper uses k=60 for best results:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        """
        k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
        fused_scores = {}
        text_to_node = {}
        try:
            
            for nodes_with_scores in results_dict.values():
                for rank, node_with_score in enumerate(
                    sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
                ):
                    text = node_with_score.node.get_content()
                    text_to_node[text] = node_with_score
                    if text not in fused_scores:
                        fused_scores[text] = 0.0
                    fused_scores[text] += 1.0 / (rank + k)
                    
            reranked_results = dict(
                sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            )

            reranked_nodes: List[NodeWithScore] = []
            for text, score in reranked_results.items():
                reranked_nodes.append(text_to_node[text])
                reranked_nodes[-1].score = score

            return reranked_nodes[:similarity_top_k]
        except Exception as e:
                print(f"Error occurred while Fusing Results: {e}")


        
    async def run_queries(self,queries, retrievers):
        print("Running Queries...")
        tasks = []
        for query in queries:
            for retriever in retrievers:
                tasks.append(retriever.aretrieve(query))

        task_results = await asyncio.gather(*tasks)

        results_dict = {}
        for query, query_result in zip(queries, task_results):
            results_dict[query] = query_result

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
        
        if(self.generate_queries_flag):
            # Generate additional queries with the help of llm
            queries = generate_queries(self.query_gen_prompt, self._llm, query_bundle.query_str, num_queries=4)
            self.generated_queries = queries  # Store generated queries
            results =  await self.run_queries(self.generated_queries, self._retrievers)
            final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
        else:
            queries = [query_bundle.query_str]
            self.generated_queries = queries  # Store the main query
            results =  await self.run_queries(self.generated_queries, self._retrievers)
            final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
       
        return final_results
