from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from typing import List
from backend.llm import fuse_results, generate_queries
from evaluation.deep_eval import deep_evaluate
from evaluation.tonic_validate import store_and_upload_results

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
        
    async def run_queries(self,queries, retrievers):
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
        print("Insiude _retrieve")
        if(not self.generate_queries_flag):
            self.generated_queries = generate_queries(self.query_gen_prompt, self._llm, query_bundle.query_str, num_queries=6)
            print(self.generated_queries)
        results =  asyncio.run(self.run_queries(self.generated_queries, self._retrievers))
        final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
        return final_results
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        print("Insiude _aretrieve")
        print("Query Generation: ", self.generate_queries_flag)
        
        if(self.generate_queries_flag):
            # Generate additional queries with the help of llm
            queries = generate_queries(self.query_gen_prompt, self._llm, query_bundle.query_str, num_queries=4)
            self.generated_queries = queries  # Store generated queries
            print(self.generated_queries)
            results =  await self.run_queries(self.generated_queries, self._retrievers)
            final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
        else:
            queries = [query_bundle.query_str]
            self.generated_queries = queries  # Store the main query
            results =  await self.run_queries(self.generated_queries, self._retrievers)
            final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
       
        return final_results
