import os

PERSIST_DIR = "persisted_index"
DATA_DIR = "data/240605_001_relevant_scopusdata.xlsx"
QUERY_GEN_PROMPT = """
    You are assisting in generating multiple search queries tailored for a RAG system 
    focused on translating technological concept search queries into bio-mimicry or bio specific language. 
    Translate the following technical query into biomimicry or biological queries by translating keywords or key concepts taken 
    from the original query and using their synonyms that are mostly used to convey the same concept in biological publications. 
    The goal here to achieve is to help the engineers to get inspiration from nature and use that to design the systems that are 
    to be used by humans.
    Generate {num_queries} search queries, one on each line,
    related to the following input query:\n"
    Query: {query}\n"
    Queries:\n"
"""

QA_PROMPT = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: \

"""

REFINE_PROMPT = """\
The original query is as follows: {query_str}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer 
(only if needed) with some more context below. 
------------
{context_str}
------------
Given the new context, refine the original answer to better answer the query. 
If the context isn't useful, return the original answer. The final answer should be comprehensive and in-depth. 
It can go up to 5-600 words if there is enough context there to write on.
Refined Answer: 
"""