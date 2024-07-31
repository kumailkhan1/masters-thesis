import asyncio
from pymongo import MongoClient
from llm import query_llm
from evaluation.deep_eval import deep_evaluate
import datetime

async def run_experiment(queries):
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['llm_evaluation']
    collection = db['results']

    # Loop through all queries
    for query in queries:
        try:
            # Run query and get response
            data = await query_llm(query)
            response = data['response']
            retrieved_nodes = data['retrieved_nodes']
            
            # Evaluate and store results
            print("Evaluating response...")
            metrics_scores = await deep_evaluate(query, response, retrieved_nodes)
            print(f"Results for '{query}': {metrics_scores}")

            # Store detailed results in MongoDB
            collection.insert_one({
                "query": query,
                "response": response,
                "context_source": retrieved_nodes,
                "evaluation_scores": metrics_scores,
                "timestamp": datetime.datetime.now(),
            })
        except Exception as e:
            print(f"Error occurred for query '{query}': {e}")

# Define your queries
queries = [
    # List your 25 queries here
]

# Run the experiment
asyncio.run(run_experiment(queries))
