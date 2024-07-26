import os

PERSIST_DIR = "persisted_index"
DATA_DIR = "data/240605_001_relevant_scopusdata.xlsx"
QUERY_GEN_PROMPT = """
    You are assisting in generating multiple search queries tailored for a RAG system focused on translating technological concept search queries into biomimicry or biologically specific language. 
    Convert the following technical query into concise biomimicry or biological queries, using keywords or key concepts from the original query and their synonyms commonly found in biological publications. 
    Ensure each query is distinct, contains fewer than 9 words, and avoids overlapping terms. 
    The goal here to achieve is to help the engineers to get inspiration from nature and use that to design the systems that are to be used by humans.
    Generate {num_queries} search queries, one on each line,
    related to the following input query:\n"
    Query: {query}\n"
    Queries:\n"
"""

QA_PROMPT = """

Based solely on the context information provided below, answer the query. Do not include prior knowledge or unrelated information. 
Ensure the answer is concise and directly addresses the query, ideally within 5-600 words.
Context:
---------------------
{context_str}
---------------------
Query: {query_str}
Answer: \

"""

REFINE_PROMPT = """\
The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
Refine the existing answer using the new context provided below. 
Only refine, extend or change the answer if the context adds significant value; otherwise, keep the original answer. 
Do not mention in the response that the answer has been refined with additional context. The final refined answer must be a stand-alone answer.
The final answer should be concise, roughly 6-700 words but not more than 800 words, and free from any inaccuracies or unrelated information.
Do not include prior knowledge or unrelated information and also do not cite sources outside the given context.
------------
{context_str}
------------
Refined Answer: 
"""