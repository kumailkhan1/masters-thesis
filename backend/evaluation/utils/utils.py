import datetime
from pymongo import MongoClient


def save_results(query_str,response,retrieved_nodes_titles,metrics_scores,generated_queries,table_name):
    
    # MongoDB setup
    client = MongoClient('mongodb://localhost:27017/')
    db = client['llm_evaluation']
    collection = db[table_name]
    # Store results in MongoDB
    result_data = {
        "query": query_str,
        "response": response,
        "context_source": retrieved_nodes_titles,
        "evaluation_scores": metrics_scores,
        "timestamp": datetime.datetime.now(),
        "generated_queries":generated_queries
    }
        
    print("Storing results in db...")
    collection.insert_one(result_data)
    