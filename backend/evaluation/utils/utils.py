import datetime
from pymongo import MongoClient
import numpy as np


def save_results(query_str, response, retrieved_nodes_titles, metrics_scores, generated_queries, table_name):
    
    # MongoDB setup
    client = MongoClient('mongodb://localhost:27017/')
    db = client['MixedBread']
    collection = db[table_name]

    # Convert numpy float32 to native Python float
    for node in retrieved_nodes_titles:
        if isinstance(node['score'], np.float32):
            node['score'] = float(node['score'])

    # Store results in MongoDB
    result_data = {
        "query": query_str,
        "response": response,
        "context_source": retrieved_nodes_titles,
        "evaluation_scores": metrics_scores,
        "timestamp": datetime.datetime.now(),
        "generated_queries": generated_queries
    }
        
    print("Storing results in db...")
    collection.insert_one(result_data)
