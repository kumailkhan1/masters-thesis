import os
from llama_index.core import (
    VectorStoreIndex, Document,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core import StorageContext, load_index_from_storage
from typing import List
import pandas as pd

from backend.retrievers import FusionRetriever
import config


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
        print("Creating an index...")
        df = pd.read_excel(data_path, header=0)
        documents = create_documents(df)
        index = VectorStoreIndex.from_documents(
            documents, embed_model=embed_model
        )
        index.storage_context.persist(persist_dir=persist_index_path)
    else:
        print("Loading the index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_index_path)
        index = load_index_from_storage(storage_context)

    return index

def generate_queries(query_gen_prompt, llm, query_str: str, num_queries: int):
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries - 1, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    # Appending the original query too
    queries.append(query_str)
    return queries

def fuse_results(results_dict, similarity_top_k: int = 5):
    # for 4 queries total of 80 documents from both the retrievers would be returned so the max_rank would be 80
    # settings k = max_rank so it is easier to interpret
    k = 80.0 
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

    return cur_response, fmt_prompts
