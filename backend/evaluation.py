from llama_index.evaluation.tonic_validate import (
    TonicValidateEvaluator, AnswerConsistencyEvaluator,
    AugmentationAccuracyEvaluator, AugmentationPrecisionEvaluator, RetrievalPrecisionEvaluator
)
from tonic_validate import ValidateApi,Run, RunData
import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from tonic_validate.services.openai_service import OpenAIService

import tiktoken

# Load environment variables from .env file
load_dotenv()
# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['llm_evaluation']
collection = db['results']

async def store_and_upload_results(query, response, retrieved_nodes):
    print("--->Inside evaluation...")

    try:
        # Prepare the data
        retrieval_context = [node.get_content() for node in retrieved_nodes]
        retrieved_nodes_titles = [
            {"title": node.node.metadata.get('Title', 'No title'), "score": node.score}
            for node in retrieved_nodes
        ]
        
        print("--->Data ready...")
        # Initialize the encoder for the model
        encoder = tiktoken.encoding_for_model("gpt-4o")
        openai_service = OpenAIService(model="gpt-4o",encoder=encoder)
        # Individual metric evaluations
        metrics = {
            "answer_consistency": AnswerConsistencyEvaluator(openai_service),         #whether the answer has information that does not appear in the retrieved context
            "augmentation_accuracy": AugmentationAccuracyEvaluator(openai_service),   #measeures the percentage of the retrieved context that is in the answer
            "augmentation_precision": AugmentationPrecisionEvaluator(openai_service), #measures whether the relevant retrieved context makes it into the answer
            "retrieval_precision": RetrievalPrecisionEvaluator(openai_service)        #measures the percentage of retrieved context is relevant to answer the question
        }
        scores = {}
        run_data_list = []
        for metric_name, evaluator in metrics.items():
            result = await evaluator.aevaluate(query=query, response=response, contexts=retrieval_context)
            scores[metric_name] = result.score
              # Prepare RunData for each evaluation
            run_data = RunData(
                scores={metric_name: result.score},
                reference_question=query,
                reference_answer=None,
                llm_answer=response,
                llm_context=retrieval_context,
            )
            run_data_list.append(run_data)

        print("--->Evaluation done...")
        print(scores)  # Log the evaluation results
        
        # Create Run object
        run = Run(
            overall_scores=scores,
            run_data=run_data_list,
            id=None
        )

        # Store results in MongoDB
        result_data = {
            "query": query,
            "response": response,
            "context_source": retrieved_nodes_titles,
            "evaluation_scores": scores,
            "timestamp": datetime.datetime.now()
        }
        collection.insert_one(result_data)
        
        # Upload results to TonicValidate UI
        validate_api = ValidateApi(api_key=os.getenv("TONIC_VALIDATE_API_KEY"))
        project_id = os.getenv("TONIC_UI_PROJECT_ID")
        validate_api.upload_run(project_id, run)
        
        print("--->Results stored and uploaded.")
    
    except Exception as e:
        print(f"Error occurred: {e}")
