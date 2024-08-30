from dotenv import load_dotenv
import os
import sys

# Load environment variables from .env file
load_dotenv()

# Add the PYTHONPATH from the .env file
sys.path.append(os.getenv("PYTHONPATH"))

import asyncio
from retrievers.utils.strategy.create_and_refine import generate_response_cr
from llm import query_llm
from evaluation.deep_eval import deep_evaluate
from retrievers.utils.utils import get_or_build_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from retrievers.FusionRetriever import FusionRetriever
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings,PromptTemplate
import config

TABLE_NAME_BM25 = "bm25_experiment_results"
TABLE_NAME_VECTOR = "vector_experiment_results"
TABLE_NAME_HYBRID = "hybrid2_experiment_results"

# Define the datasets with corresponding paper titles and queries
# datasets = {
#     "benchmark_dataset_1": {
#         "title": "Functional morphology of the fin rays of teleost fishes",
#         "queries": {
#             "KQ": "Adaptive fin-like mechanisms in robotics for enhanced manoeuvrability",
#             "AQ": "Develop a robotic gripper that adapts to the shape of objects for a secure and gentle grip."
#         }
#     },
#     "benchmark_dataset_2": {
#         "title": "Mechanoresponsive self-growing hydrogels inspired by muscle training",
#         "queries": {
#             "KQ": "What are potential applications for materials that can inhibit self-growing capabilities",
#             "AQ": "Design a smart material that can make itself stronger with each use"
#         }
#     },
#     "benchmark_dataset_3": {
#         "title": "A contribution to the functional analysis of the foot of the Tokay, Gekko gecko (Reptilia: Gekkonidae)",
#         "queries": {
#             "KQ": "Engineer a adhesive material that can stick to various surfaces repeatedly without losing its stickiness",
#             "AQ": "Development of synthetic gripping pads for vertical motion in robotics"
#         }
#     },
#     "benchmark_dataset_4": {
#         "title": "The salvinia paradox: superhydrophobic surfaces with hydrophilic pins for air retention under water",
#         "queries": {
#             "KQ": "Design a hydrophobic gel for cars",
#             "AQ": "Create a coating that reduces drag and enhance buoyancy for boats."
#         }
#     },
#     "benchmark_dataset_5": {
#         "title": "Plant Cuticles: Physicochemical Characteristics and Biosynthesis",
#         "queries": {
#             "KQ": "Inspired by nature, how can we create a material that offers waterproofing capabilities?",
#             "AQ": "Design a self-cleaning surface for outdoor equipment that repels dirt and water effectively."
#         }
#     }
# }


datasets = {
    "benchmark_dataset_1": {
        "title": "Functional morphology of the fin rays of teleost fishes",
        "queries": [
            "Develop a robotic gripper that adapts to the shape of objects for a secure and gentle grip.",
            "Design structure that can automatically adapt to changing loads.",
            "Design of prosthetic limbs that need to adapt to different surfaces or forces.",
            "Structural innovations for controlled deformation in high-stress conditions.",
            "Adaptive segmented materials that optimize flexibility and strength."
        ]
    },
    "benchmark_dataset_2": {
        "title": "Mechanoresponsive self-growing hydrogels inspired by muscle training",
        "queries": [
            "What are potential applications for materials that can grow stronger with use.",
            "Materials that fortify under cyclic mechanical influence.",
            "Smart materials that self-strengthen in response to any force applied.",
            "Substances that reinforce themselves through repetitive mechanical influence.",
            "How can mechanical stress-induced growth in materials contribute to advancements in adaptive architecture?"
        ]
    },
    "benchmark_dataset_3": {
        "title": "A contribution to the functional analysis of the foot of the Tokay, Gekko gecko (Reptilia: Gekkonidae)",
        "queries": [
            "Engineer a material that can stick to various surfaces repeatedly without losing its stickiness.",
            "Development of synthetic gripping pads for vertical locomotion in robotics.",
            "Nature-inspired sticky surfaces with improved resilience and versatility.",
            "Evolutionary innovations in organism-surface interactions for vertical mobility.",
            "Integrating sensory feedback for precise limb placement in autonomous systems."
        ]
    },

    "benchmark_dataset_4": {
        "title": "The salvinia paradox: superhydrophobic surfaces with hydrophilic pins for air retention under water",
        "queries": [
            "Create a surface coating that reduces drag and enhance buoyancy for boats.",
            "Dual-nature surfaces for controlling fluid-solid interactions in undersea conditions.",
            "How can natural interfaces that balance the fluid influence the design of new underwater technologies?",
            "Surface engineering to prevent ice formation on aircraft wings.",
            "Smart fabrics that regulate moisture and air content for improved comfort."
        ]
    },
    "benchmark_dataset_5": {
        "title": "Plant Cuticles: Physicochemical Characteristics and Biosynthesis",
        "queries": [
            "Design a self-cleaning surface for outdoor equipment that repels dirt and water effectively.",
            "Create a dirt-repelling and water repelling gel that can be applied on cars.",
            "Design a nanostructured surface for wind turbines that prevents dirt build-up and reduces maintenance.",
            "Adaptive exterior coatings that respond to atmospheric changes.",
            "Nano-scale natural defenses that inspire technological breakthroughs."
        ]
    },
}


async def main():
    Settings.llm = Ollama(model="mistral", request_timeout=60.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    for dataset_name, details in datasets.items():
        print(f"Processing {dataset_name} with title: {details['title']}")
        # Set up the index for each dataset
        persist_dir = f'data/benchmark/persisted_index_{dataset_name}'
        index = await get_or_build_index(Settings.embed_model, persist_dir=persist_dir, data_dir=f'data/benchmark/{dataset_name}.csv')

        # Prepare the retrievers
        bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)
        vector_retriever = index.as_retriever(similarity_top_k=10)
        fusion_retriever = FusionRetriever(Settings.llm, config.QUERY_GEN_PROMPT, [vector_retriever,bm25_retriever], similarity_top_k=5,generate_queries_flag=False)
   
        qa_prompt = PromptTemplate(config.QA_PROMPT)
        refine_prompt = PromptTemplate(config.REFINE_PROMPT)
                
        # Run experiments for each query
        for query_str in details['queries']:   
            retrieved_nodes = await fusion_retriever.aretrieve(query_str)
            response, fmt_prompts = await generate_response_cr(retrieved_nodes, query_str, qa_prompt, refine_prompt, Settings.llm) #Create and Refine
            await deep_evaluate(query_str,str(response),retrieved_nodes,fusion_retriever.generated_queries,TABLE_NAME_HYBRID) #DeepEval
           
# Run the experiment
asyncio.run(main())
