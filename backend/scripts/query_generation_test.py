import os
import json
import pandas as pd

# Get the current working directory and navigate accordingly
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')

# Load the data
file_path = os.path.join(data_dir, 'llm_evaluation.Query_Generation_Experiment.json')
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define the mapping of papers to queries
paper_query_mapping = {
    "Natural and synthetic superhydrophobic surfaces: A review of the fundamentals, structures, and applications": [
        "How can self-cleaning surfaces inspired by lotus leaves be applied in solar panel maintenance?",
        "What natural processes can be used to design surfaces that repel water and oil?",
        "Explore the application of hydrophobic coatings in reducing equipment cleaning costs.",
        "What are the differences between hydrophilic and hydrophobic surfaces in self-cleaning technology?",
        "How do butterfly wings inspire the design of water-repellent surfaces for industrial use?"
    ],
    "A distributed algorithm to maintain and repair the trail networks of arboreal ants": [
        "How can distributed algorithms from nature be used to maintain and repair network infrastructure autonomously?",
        "What lessons from turtle ant trail networks can optimize traffic flow in urban planning?",
        "How can swarm intelligence models mimic ant behavior for efficient resource routing in logistics?",
        "What biological principles can improve fault-tolerance in robotic systems inspired by ant trail maintenance?",
        "How can natural algorithms from ants inform the design of resilient communication networks?"
    ]
}

# Function to check if the relevant paper is found in the context source and return its position and evaluation scores
def check_paper_in_context(query, data, relevant_paper):
    results = []
    for item in data:
        if item['query'] == query:
            for index, source in enumerate(item['context_source']):
                if relevant_paper in source['title']:
                    query_generation_enabled = len(item['generated_queries']) > 1
                    results.append({
                        'query': query,
                        'paper_title': relevant_paper,
                        'position_with_paper': index + 1,
                        'answer_relevancy': item['evaluation_scores'].get('answer_relevancy', None),
                        'faithfulness': item['evaluation_scores'].get('faithfulness', None),
                        'contextual_relevancy': item['evaluation_scores'].get('contextual_relevancy', None),
                        'query_generation_enabled': query_generation_enabled,
                        'generated_queries': item['generated_queries']
                    })
    return results

final_results = []

# Check each paper and its associated queries
for paper, queries in paper_query_mapping.items():
    paper_results = []
    for query in queries:
        results = check_paper_in_context(query, data, paper)
        final_results.extend(results)
        paper_results.extend(results)

    # Calculate averages for the paper
    if paper_results:
        df_paper = pd.DataFrame(paper_results)
        avg_answer_relevancy_false = df_paper[df_paper['query_generation_enabled'] == False]['answer_relevancy'].mean()
        avg_faithfulness_false = df_paper[df_paper['query_generation_enabled'] == False]['faithfulness'].mean()
        avg_contextual_relevancy_false = df_paper[df_paper['query_generation_enabled'] == False]['contextual_relevancy'].mean()

        avg_answer_relevancy_true = df_paper[df_paper['query_generation_enabled'] == True]['answer_relevancy'].mean()
        avg_faithfulness_true = df_paper[df_paper['query_generation_enabled'] == True]['faithfulness'].mean()
        avg_contextual_relevancy_true = df_paper[df_paper['query_generation_enabled'] == True]['contextual_relevancy'].mean()

        final_results.append({
            'query': 'Average Scores for Paper (Query Gen False)',
            'paper_title': paper,
            'position_with_paper': None,
            'answer_relevancy': avg_answer_relevancy_false,
            'faithfulness': avg_faithfulness_false,
            'contextual_relevancy': avg_contextual_relevancy_false,
            'query_generation_enabled': False,
            'generated_queries': None
        })

        final_results.append({
            'query': 'Average Scores for Paper (Query Gen True)',
            'paper_title': paper,
            'position_with_paper': None,
            'answer_relevancy': avg_answer_relevancy_true,
            'faithfulness': avg_faithfulness_true,
            'contextual_relevancy': avg_contextual_relevancy_true,
            'query_generation_enabled': True,
            'generated_queries': None
        })

# Convert the results to a DataFrame
df_results = pd.DataFrame(final_results)

# Save the results to a CSV file
output_file = os.path.join(current_dir, '..', 'data', 'query_generation_experiment_results_with_averages.csv')
df_results.to_csv(output_file, index=False)

print(f"Analysis results have been saved to {output_file}")
