import os
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Document, Settings,
    get_response_synthesizer, ServiceContext, PromptTemplate, QueryBundle
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from typing import List
from tqdm.asyncio import tqdm
import pandas as pd

import config

import asyncio
def create_documents(df):
    df.drop(['Author full names', 'Author(s) ID', 'Page start', 'Page end', 'Page count'], axis=1, inplace=True)
    df = df[df['Abstract'] != '[No abstract available]']
    documents = []
    for _, row in df.iterrows():
        metadata = row.drop('Abstract').to_dict()
        doc = Document(text=row['Abstract'], metadata=metadata)
        documents.append(doc)
    return documents

async def get_or_build_index(embed_model, persist_dir=config.PERSIST_DIR, data_dir=config.DATA_DIR):
    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_dir)
    persist_index_path = os.path.join(cwd, persist_dir)

    if not os.path.exists(persist_index_path):
        df = pd.read_excel(data_path, header=0)
        documents = create_documents(df)
        index = VectorStoreIndex.from_documents(
            documents, embed_model=embed_model
        )
        index.storage_context.persist(persist_dir=persist_index_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_index_path)
        index = load_index_from_storage(storage_context)

    return index

def generate_queries(query_gen_prompt, llm, query_str: str, num_queries: int = 6):
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    # Appending the original query too
    queries.append(query_str)
    return queries

def fuse_results(results_dict, similarity_top_k: int = 5):
    k = 60.0
    fused_scores = {}
    text_to_node = {}

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

class FusionRetriever(BaseRetriever):
    def __init__(
        self,
        llm,
        query_gen_prompt,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        self._llm = llm
        self.query_gen_prompt = query_gen_prompt
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
        queries = generate_queries(self.query_gen_prompt, self._llm, query_bundle.query_str, num_queries=4)
        print(queries)
        results =  asyncio.run(self.run_queries(queries, self._retrievers))
        final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
        return final_results
    
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        print("Insiude _aretrieve")
        queries = generate_queries(self.query_gen_prompt, self._llm, query_bundle.query_str, num_queries=4)
        print(queries)
        results =  await self.run_queries(queries, self._retrievers)
        final_results = fuse_results(results, similarity_top_k=self._similarity_top_k)
        return final_results

async def generate_response_cr(
    retrieved_nodes, query_str, qa_prompt, refine_prompt, llm
):
    cur_response = None
    fmt_prompts = []
    for idx, node in enumerate(retrieved_nodes):
        context_str = node.get_content()
        if idx == 0:
            fmt_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
        else:
            fmt_prompt = refine_prompt.format(
                context_str=context_str,
                query_str=query_str,
                existing_answer=str(cur_response),
            )

        cur_response = llm.complete(fmt_prompt)
        fmt_prompts.append(fmt_prompt)

    return str(cur_response), fmt_prompts

async def query_llm(query_str):
    Settings.llm = Ollama(model="mistral", request_timeout=60.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = await get_or_build_index(embed_model=Settings.embed_model)

    query_gen_prompt = PromptTemplate(config.QUERY_GEN_PROMPT)
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

    fusion_retriever = FusionRetriever(
        Settings.llm, query_gen_prompt, [vector_retriever, bm25_retriever], similarity_top_k=5
    )
    retrieved_nodes = await fusion_retriever.aretrieve(query_str)
    
    qa_prompt = PromptTemplate(config.QA_PROMPT)
    refine_prompt = PromptTemplate(config.REFINE_PROMPT)
    response, fmt_prompts = await generate_response_cr(retrieved_nodes, query_str, qa_prompt, refine_prompt, Settings.llm)
    
    
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
        "response":response,
        "retrieved_nodes":extracted_data
    }
    return data
