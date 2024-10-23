import datetime
from pymongo import MongoClient
import numpy as np
from pydantic import BaseModel


def save_results(query_str, response, retrieved_nodes, metrics_scores, generated_queries, table_name):
    
    # MongoDB setup
    client = MongoClient('mongodb://localhost:27017/')
    db = client['SenckenBerg']
    collection = db[table_name]

    # Convert numpy float32 to native Python float
    for node in retrieved_nodes:
        if isinstance(node.score, np.float32):
            node.score = float(node.score)

    # Store results in MongoDB
    result_data = {
        "query": query_str,
        "response": str(response),
        "context_source": [
                {
                "metadata":{
                    "Title": node.node.metadata.get("Title", "No title"),
                    "DOI": node.node.metadata.get("DOI", "No DOI"),
                    "Authors": node.node.metadata.get("Authors", "No authors"),
                    "Abstract": node.node.text
                },
                "score":node.score
            } for node in retrieved_nodes
        ],

        "evaluation_scores": metrics_scores,
        "timestamp": datetime.datetime.now(),
        "generated_queries": generated_queries
    }
        
    print("Storing results in db...")
    collection.insert_one(result_data)
