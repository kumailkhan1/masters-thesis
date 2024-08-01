import asyncio
from pymongo import MongoClient
from llm import query_llm
from evaluation.deep_eval import deep_evaluate
import datetime

async def run_experiment(queries):
    # Loop through all queries
    for query in queries:
        try:
            # Running Query
            print("="*10)
            print("Query: ",query)
            print("="*10)
            # Run query and get response
            data = await query_llm(query)
            response = data['response']
            retrieved_nodes = data['retrieved_nodes']
            generated_queries = data['generated_queries'] 
            
            # Evaluate and store results
            print("Evaluating response...")
            metrics_scores = await deep_evaluate(query, response, retrieved_nodes,generated_queries,"RetrievalExperiment_with_papers")
            print(f"Results for '{query}': {metrics_scores}")
            
        except Exception as e:
            print(f"Error occurred for query '{query}': {e}")

queries = [
"How can slime mold-inspired models help in understanding the distribution of dark matter?",
"What biological principles can improve fault-tolerance in robotic systems inspired by ant trail maintenance?",
"What natural mechanisms can inform the development of adaptive robotic grippers for delicate tasks?"
]


# Run the experiment
asyncio.run(run_experiment(queries))
