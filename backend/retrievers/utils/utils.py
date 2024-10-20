import os
from llama_index.core import (
    VectorStoreIndex, Document,
)
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.core import StorageContext, load_index_from_storage
from typing import List
import pandas as pd

import config
import chardet

def create_documents(df):
    df = df[df['Abstract'] != '[No abstract available]']
    documents = []
    for _, row in df.iterrows():
        # metadata = row[['Title','Authors', 'DOI', 'Link', 'Author Keywords']].to_dict()
        metadata = row[['Title','Authors',]].to_dict()
        
        doc = Document(text=row['Abstract'],
                       metadata=metadata,)
                    #    excluded_llm_metadata_keys=['Authors', 'DOI', 'Link', 'Author Keywords'], # excludes these cols from LLM response
                    #    excluded_embed_metadata_keys=[['Authors', 'DOI', 'Link']]) # excludes these from embedding
        documents.append(doc)
    return documents

async def get_or_build_index(embed_model, persist_dir=config.PERSIST_DIR, data_dir=config.DATA_DIR):

    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_dir)
    persist_index_path = os.path.join(cwd, persist_dir)

    with open(data_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    print(f"The detected encoding is: {encoding}")
    if not os.path.exists(persist_index_path):
        print("Creating an index...")
        df = pd.read_csv(data_path, header=0)
        documents = create_documents(df)
        try:
            # Create a custom text splitter with a larger chunk size
            text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=100)
            
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model,
                transformations=[text_splitter],
                show_progress=True
            )
        except Exception as e:
            print(f"Failed to create index: {e}")
            index = None
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

def fuse_results(results_dict, similarity_top_k):
    print("Fusing results...")
    # for 4 queries total of 80 documents (if top k = 10 for both retrievers) from both the retrievers would be returned so the max_rank would be 80
    # settings k = max_rank so it is easier to interpret
    k = 80
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
                
        # # Normalize the scores to ensure they stay within 0-1
        # max_score = max(fused_scores.values(), default=1)
        # for text in fused_scores:
        #     fused_scores[text] /= max_score  # Normalize to 0-1 range
            
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

