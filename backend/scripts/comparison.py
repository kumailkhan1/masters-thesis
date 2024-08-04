import os
import json

import pandas as pd


# Get the current working directory and navigate accordingly
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')

file_with_papers = os.path.join(data_dir, 'llm_evaluation.random_papers_experiment.json')
# file_without_papers = os.path.join(data_dir, 'llm_evaluation.RetrievalExperiment_without_papers.json')

# Load the data
with open(file_with_papers, 'r',encoding='utf-8') as f:
    data_with_papers = json.load(f)

# with open(file_without_papers, 'r',encoding='utf-8') as f:
#     data_without_papers = json.load(f)

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
    ],
    "Bioinspiration and Biomimetic Art in Robotic Grippers": [
        "How can biomimetic gripper designs improve autonomous object manipulation in robotics?",
        "What natural mechanisms can inform the development of adaptive robotic grippers for delicate tasks?",
        "How can animal-inspired gripping techniques enhance robotic dexterity in manufacturing?",
        "What materials and actuation methods from nature can optimize the efficiency of robotic grippers?",
        "How can bioinspired grippers be utilized to improve automation in healthcare and service industries?"
    ],
    "Revealing the Dark Threads of the Cosmic Web": [
        "How can biomimetic algorithms enhance the mapping of dark matter in the cosmic web?",
        "What role does the intergalactic medium play in the structure of the cosmic web, according to new detection methods?",
        "How can slime mold-inspired models help in understanding the distribution of dark matter?",
        "What are the implications of shock-heating and ionization in cosmic web filaments for cosmological studies?",
        "How does the new method for detecting cosmic web structures compare to traditional spectroscopy techniques in revealing hidden matter?"
    ],
    "Mechanoresponsive self-growing hydrogels inspired by muscle training": [
        "How can synthetic materials emulate the adaptive growth and strengthening observed in biological tissues?",
        "What principles from muscle adaptation can be applied to develop self-repairing materials?",
        "How can we create polymeric systems that enhance durability through repeated mechanical stimuli?",
        "What are potential applications for materials that can grow stronger with use, inspired by natural processes?",
        "How can engineered hydrogels be designed to mimic the self-repairing capabilities of living organisms?"
    ]
}

# Function to check if the relevant paper is found in the context source and return its position
def check_paper_in_context(query, data, relevant_paper):
    for item in data:
        if item['query'] == query:
            for index, source in enumerate(item['context_source']):
                if relevant_paper in source['title']:
                    return index
    return None

results = []

# Check each paper and its associated queries
for paper, queries in paper_query_mapping.items():
    for query in queries:
        with_paper_position = check_paper_in_context(query, data_with_papers, paper)
        # without_paper_position = check_paper_in_context(query, data_without_papers, paper)
        results.append({
            'query': query,
            'paper_title': paper,
            'retrieved_with_paper': with_paper_position is not None,
            'position_with_paper': with_paper_position + 1,
            # 'retrieved_without_paper': without_paper_position is not None,
            # 'position_without_paper': without_paper_position
        })

# Convert the results to a DataFrame
df_results = pd.DataFrame(results)

# Save the results to a CSV file
output_file = os.path.join(current_dir, '..', 'data', 'random_papers_experiment_results.csv')
df_results.to_csv(output_file, index=False)

print(f"Comparison results have been saved to {output_file}")