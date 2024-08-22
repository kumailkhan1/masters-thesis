import asyncio
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric,ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
# Load environment variables from .env file
from dotenv import load_dotenv

from evaluation.utils.utils import save_results
load_dotenv()

EVAL_MODEL = "gpt-3.5-turbo"

async def deep_evaluate(query_str,response,retrieved_nodes,generated_queries,table_name):
    try:
        retrieved_nodes_titles = [
                {"title": node.node.metadata.get('Title', 'No title'), "score": node.score}
                for node in retrieved_nodes
            ]
        retrieval_context = [node.get_content() for node in retrieved_nodes]
        
        # The answer relevancy metric measures the quality of your RAG pipeline's generator 
        # by evaluating how relevant the actual_output of your LLM application is compared to the provided input
        answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.6,
            model=EVAL_MODEL,
            include_reason=True,
            verbose_mode=False
        )
        #The faithfulness metric measures the quality of your RAG pipeline's generator by evaluating whether 
        # the actual_output factually aligns with the contents of your retrieval_context. 
        faithfulness_metric = FaithfulnessMetric(
            threshold=0.6,
            model=EVAL_MODEL,
            include_reason=True,
            verbose_mode=False
        )
        # The contextual relevancy metric measures the quality of your RAG pipeline's retriever by evaluating 
        # the overall relevance of the information presented in your retrieval_context for a given input
        contextual_relevancy_metric = ContextualRelevancyMetric(
            threshold=0.6,
            model=EVAL_MODEL,
            include_reason=True,
            verbose_mode=False
        )
        without_context_test_case = LLMTestCase(
            input=query_str,
            actual_output=response
        )

        with_context_test_case = LLMTestCase(
            input=query_str,
            actual_output=response,
            retrieval_context=retrieval_context
        )
        print("Running Metrics...")
        results = await asyncio.gather(
            answer_relevancy_metric.a_measure(without_context_test_case),
            faithfulness_metric.a_measure(with_context_test_case),
            contextual_relevancy_metric.a_measure(with_context_test_case)
        )
        metrics_scores = {
        "answer_relevancy": results[0],
        "faithfulness": results[1],
        "contextual_relevancy": results[2],
    }
        # Save to db
        save_results(query_str,response,retrieved_nodes_titles,metrics_scores,generated_queries,table_name)
        print(metrics_scores)
        return metrics_scores
    except Exception as e:
            print(f"Error occurred: {e}")
            
            
