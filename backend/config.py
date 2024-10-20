import os

PERSIST_DIR = "data/persisted_index_senckenberg"
DATA_DIR = "data/Senckenberg_Paper_Scopus_240618.csv"
QUERY_GEN_PROMPT = """
You are assisting in generating multiple search queries tailored for a Retrieval-Augmented Generation (RAG) system 
that helps translate a user's query into biomimicry or biology-specific language. The user has no prior knowledge of 
biology or biomimicry, and their focus is on innovation—finding solutions in nature (animals, plants, microorganisms, 
ecosystems) that can inspire new technologies or improvements.

For each query:
- Expand the original query semantically, by considering natural analogies and solutions inspired by nature.
- Avoid repetition of the same keywords from the original query.
- Focus on diversity in the generated queries, ensuring that each query reflects a different aspect of biology or biomimicry.
- Each query should be concise (10-12 words) and clearly focused on translating the engineer's technical needs into biologically relevant concepts.

Additional Guidelines:
- Provide distinct keywords in each query to help retrieve different kinds of biological solutions.
- Contextual Expansion: Keep the intent of solving problems through nature in mind, such as using biological strategies to address engineering challenges.

Below are a few examples of how queries are translated. Follow the same pattern for expanding any new query.

Few-shot Examples:

Original Query: "Design a hydrodynamic fuselage coating for large ocean-going tankers to decrease water resistance and enhance fuel economy."
Expanded Queries:
1. "Explore marine animals' skin adaptations for reducing drag in large vessels."
2. "Biomimetic hydrodynamic surface patterns inspired by shark skin to reduce water drag."
3. "Nature-based water-resistant coatings for energy-efficient transportation systems."

Original Query: "Engineer a load-bearing composite material that exhibits a distinct color shift upon exceeding its specified load capacity."
Expanded Queries:
1. "Biomaterials with color-shifting properties in response to mechanical stress in nature."
2. "Explore biological structures with adaptive color-changing under physical load."
3. "Natural mechanisms for material strength indicators in load-bearing systems."

Original Query: "Develop fabric that dynamically alters its thermal conductivity in response to varying environmental conditions."
Expanded Queries:
1. "Biological systems that adjust thermal conductivity to environmental changes."
2. "Fabric inspired by natural insulation mechanisms for dynamic heat control."
3. "Adaptive temperature regulation in nature for thermal conductivity applications."

Original Query: "Create a method to improve the durability and lifespan of asphalt roads in extreme weather conditions."
Expanded Queries:
1. "Natural materials with high resistance to extreme weather for surface durability."
2. "Biomimetic strategies for weather-resistant surfaces inspired by desert organisms."
3. "Nature-based solutions for long-lasting, durable surfaces under harsh conditions."

Now generate {num_queries} queries based on the following query:
Query: {query}
Queries:
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