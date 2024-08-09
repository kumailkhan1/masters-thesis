import asyncio
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
evaluation_dir = os.path.abspath(os.path.join(current_dir, '../evaluation/'))
if evaluation_dir not in sys.path:
    sys.path.append(evaluation_dir)

from llm import query_llm
from evaluation.deep_eval import deep_evaluate
TABLE_NAME = "Query_Generation_Experiment"

async def run_experiment(queries):
    # Loop through all queries
    for query in queries:
        try:
            # Running Query
            print("="*10)
            print("Query: ",query)
            print("="*10)
            # Run query and get response
            data = await query_llm(query,generate_queries_flag=True)
            response = data['response']
            retrieved_nodes = data['retrieved_nodes']
            generated_queries = data['generated_queries'] 
            
            # Evaluate and store results
            print("Evaluating response...")
            metrics_scores = await deep_evaluate(query, response, retrieved_nodes,generated_queries,TABLE_NAME)
            print(f"Results for '{query}': {metrics_scores}")
            
        except Exception as e:
            print(f"Error occurred for query '{query}': {e}")

queries = [
    "How can self-cleaning surfaces inspired by lotus leaves be applied in solar panel maintenance?",
    "What natural processes can be used to design surfaces that repel water and oil?",
    "Explore the application of hydrophobic coatings in reducing equipment cleaning costs.",
    "What are the differences between hydrophilic and hydrophobic surfaces in self-cleaning technology?",
    "How do butterfly wings inspire the design of water-repellent surfaces for industrial use?",
    "How can distributed algorithms from nature be used to maintain and repair network infrastructure autonomously?",
    "What lessons from turtle ant trail networks can optimize traffic flow in urban planning?",
    "How can swarm intelligence models mimic ant behavior for efficient resource routing in logistics?",
    "What biological principles can improve fault-tolerance in robotic systems inspired by ant trail maintenance?",
    "How can natural algorithms from ants inform the design of resilient communication networks?",
    # "How can biomimetic gripper designs improve autonomous object manipulation in robotics?",
    # "What natural mechanisms can inform the development of adaptive robotic grippers for delicate tasks?",
    # "How can animal-inspired gripping techniques enhance robotic dexterity in manufacturing?",
    # "What materials and actuation methods from nature can optimize the efficiency of robotic grippers?",
    # "How can bioinspired grippers be utilized to improve automation in healthcare and service industries?",
    # "How can biomimetic algorithms enhance the mapping of dark matter in the cosmic web?",
    # "What role does the intergalactic medium play in the structure of the cosmic web, according to new detection methods?",
    # "How can slime mold-inspired models help in understanding the distribution of dark matter?",
    # "What are the implications of shock-heating and ionization in cosmic web filaments for cosmological studies?",
    # "How does the new method for detecting cosmic web structures compare to traditional spectroscopy techniques in revealing hidden matter?",
    # "How can synthetic materials emulate the adaptive growth and strengthening observed in biological tissues?",
    # "What principles from muscle adaptation can be applied to develop self-repairing materials?",
    # "How can we create polymeric systems that enhance durability through repeated mechanical stimuli?",
    # "What are potential applications for materials that can grow stronger with use, inspired by natural processes?",
    # "How can engineered hydrogels be designed to mimic the self-repairing capabilities of living organisms?"
]


# Run the experiment
asyncio.run(run_experiment(queries))
