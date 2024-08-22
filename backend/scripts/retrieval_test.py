import asyncio
import sys
import os

# Setting up the current directory and paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adding parent and evaluation directories to sys.path for imports
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
evaluation_dir = os.path.abspath(os.path.join(current_dir, '../evaluation/'))
if evaluation_dir not in sys.path:
    sys.path.append(evaluation_dir)

# Importing necessary functions
from llm import query_llm
from evaluation.deep_eval import deep_evaluate

# Defining the table name for storing results
TABLE_NAME = "2_random_papers_experiment"

# Define the papers and their corresponding queries
papers_queries = {
    "Plant Cuticles: Physicochemical Characteristics and Biosynthesis": [
        # "Design a self-cleaning surface for outdoor equipment that repels dirt and water effectively.",
        "How can we create a dirt-repelling and water repelling agent that can be applied on cars.",
        # "Design a nanostructured surface for wind turbines that prevents dirt build-up and reduces maintenance.",
        # "Natural surface coatings inspired by plant epidermis.",
        # "Develop an anti-fouling surface for marine equipment utilizing biomimetic technologies."
    ],
    "Functional morphology of the fin rays of teleost fishes": [
        # "Adaptive fin-like mechanisms in robotics for enhanced maneuverability.",
        "Develop a robotic gripper that adapts to the shape of objects for a secure and gentle grip.",
        # "Design a structure that can automatically adapt to changing loads.",
        # "Design of prosthetic limbs that need to adapt to different surfaces or forces."
    ],
    "Sorption of oils by the nonliving biomass of a Salvinia sp.": [
        "Hierarchical surfaces in nature for reducing fluid drag.",
        "Natural models for creating superhydrophobic surfaces.",
        "Create a surface coating that reduces drag and enhance buoyancy for boats.",
        # "Water-repellent coatings inspired by natural wax structures.",
        # "Hydrophobic biomaterials for energy-efficient surface applications."
    ],
    "A contribution to the functional analysis of the foot of the Tokay, Gekko gecko (Reptilia: Gekkonidae)": [
        "Engineer a material that can stick to various surfaces repeatedly without losing its stickiness.",
        "Development of synthetic gripping pads for vertical locomotion in robotics.",
        "Nature-inspired sticky surfaces with improved resilience and versatility."
    ],
    "Mechanoresponsive self-growing hydrogels inspired by muscle training": [
        "What are potential applications for materials that can grow stronger with use.",
        "Materials that fortify under cyclic mechanical influence.",
        "Smart materials that self-strengthen in response to any force applied."
    ]
}

# Async function to run the experiment
async def run_experiment(papers_queries):
    # Loop through each paper and its queries
    for paper, queries in papers_queries.items():
        print(f"Running experiment for paper: {paper}")
        for query in queries:
            try:
                # Print the query being processed
                print("="*10)
                print("Query: ", query)
                print("="*10)
                
                # Run the query with Query Generation enabled
                data = await query_llm(query, generate_queries_flag=False)
                
                # Extracting the response, retrieved nodes, and generated queries
                response = data['response']
                retrieved_nodes = data['retrieved_nodes']
                generated_queries = data['generated_queries'] 
                
                # Evaluate and store results
                print("Evaluating response...")
                metrics_scores = await deep_evaluate(query, response, retrieved_nodes, generated_queries, TABLE_NAME)
                
                # Print the results of the evaluation
                print(f"Results for '{query}': {metrics_scores}")
                
            except Exception as e:
                print(f"Error occurred for query '{query}': {e}")

# Run the experiment with the provided queries
asyncio.run(run_experiment(papers_queries))
