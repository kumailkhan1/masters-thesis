import os
import json

import pandas as pd


# Get the current working directory and navigate accordingly
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')

file_with_papers = os.path.join(data_dir, 'llm_evaluation.3_random_papers_experiment.json')
# file_without_papers = os.path.join(data_dir, 'llm_evaluation.RetrievalExperiment_without_papers.json')

# Load the data
with open(file_with_papers, 'r',encoding='utf-8') as f:
    data = json.load(f)

# with open(file_without_papers, 'r',encoding='utf-8') as f:
#     data_without_papers = json.load(f)

# Define the mapping of papers to queries
paper_query_mapping = {
    # "Natural and synthetic superhydrophobic surfaces: A review of the fundamentals, structures, and applications": [
    #     "How can self-cleaning surfaces inspired by lotus leaves be applied in solar panel maintenance?",
    #     "What natural processes can be used to design surfaces that repel water and oil?",
    #     "Explore the application of hydrophobic coatings in reducing equipment cleaning costs.",
    #     "What are the differences between hydrophilic and hydrophobic surfaces in self-cleaning technology?",
    #     "How do butterfly wings inspire the design of water-repellent surfaces for industrial use?"
    # ],
    # "A distributed algorithm to maintain and repair the trail networks of arboreal ants": [
    #     "How can distributed algorithms from nature be used to maintain and repair network infrastructure autonomously?",
    #     "What lessons from turtle ant trail networks can optimize traffic flow in urban planning?",
    #     "How can swarm intelligence models mimic ant behavior for efficient resource routing in logistics?",
    #     "What biological principles can improve fault-tolerance in robotic systems inspired by ant trail maintenance?",
    #     "How can natural algorithms from ants inform the design of resilient communication networks?"
    # ],
    # "Bioinspiration and Biomimetic Art in Robotic Grippers": [
    #     "How can biomimetic gripper designs improve autonomous object manipulation in robotics?",
    #     "What natural mechanisms can inform the development of adaptive robotic grippers for delicate tasks?",
    #     "How can animal-inspired gripping techniques enhance robotic dexterity in manufacturing?",
    #     "What materials and actuation methods from nature can optimize the efficiency of robotic grippers?",
    #     "How can bioinspired grippers be utilized to improve automation in healthcare and service industries?"
    # ],
    # "Revealing the Dark Threads of the Cosmic Web": [
    #     "How can biomimetic algorithms enhance the mapping of dark matter in the cosmic web?",
    #     "What role does the intergalactic medium play in the structure of the cosmic web, according to new detection methods?",
    #     "How can slime mold-inspired models help in understanding the distribution of dark matter?",
    #     "What are the implications of shock-heating and ionization in cosmic web filaments for cosmological studies?",
    #     "How does the new method for detecting cosmic web structures compare to traditional spectroscopy techniques in revealing hidden matter?"
    # ],
    # "Mechanoresponsive self-growing hydrogels inspired by muscle training": [
    #     "How can synthetic materials emulate the adaptive growth and strengthening observed in biological tissues?",
    #     "What principles from muscle adaptation can be applied to develop self-repairing materials?",
    #     "How can we create polymeric systems that enhance durability through repeated mechanical stimuli?",
    #     "What are potential applications for materials that can grow stronger with use, inspired by natural processes?",
    #     "How can engineered hydrogels be designed to mimic the self-repairing capabilities of living organisms?"
    # ]
    
    "Plant Cuticles: Physicochemical Characteristics and Biosynthesis": [
        # "Design a self-cleaning surface for outdoor equipment that repels dirt and water effectively.",
        "How can we create a dirt-repelling and water repelling agent that can be applied on cars.",
        # "Design a nanostructured surface for wind turbines that prevents dirt build-up and reduces maintenance.",
        # "Natural surface coatings inspired by plant epidermis.",
        # "Develop an anti-fouling surface for marine equipment utilizing biomimetic technologies."
    ],
    "Functional morphology of the fin rays of teleost fishes": [
        # "Adaptive fin-like mechanisms in robotics for enhanced maneuverability.",
        "Develop a robotic gripper that adapts to the shape of objects for a secure and gentle grip.",
        # "Design a structure that can automatically adapt to changing loads.",
        # "Design of prosthetic limbs that need to adapt to different surfaces or forces."
    ],
    "The salvinia paradox: superhydrophobic surfaces with hydrophilic pins for air retention under water": [
        # "Hierarchical surfaces in nature for reducing fluid drag.",
        # "Natural models for creating superhydrophobic surfaces.",
        "Create a surface coating that reduces drag and enhance buoyancy for boats.",
        # "Water-repellent coatings inspired by natural wax structures.",
        # "Hydrophobic biomaterials for energy-efficient surface applications."
    ],
    "A contribution to the functional analysis of the foot of the Tokay, Gekko gecko (Reptilia: Gekkonidae)": [
        # "Engineer a material that can stick to various surfaces repeatedly without losing its stickiness.",
        "Development of synthetic gripping hands for vertical movement in robots.",
        # "Nature-inspired sticky surfaces with improved resilience and versatility."
    ],
    "Mechanoresponsive self-growing hydrogels inspired by muscle training": [
        # "What are potential applications for materials that can grow stronger with use.",
        "Design a material that can become stronger when force is applied",
        # "Smart materials that self-strengthen in response to any force applied."
    ]
}

# Function to check if the relevant paper is found in the context source and return its position and evaluation scores
# Function to check if the relevant paper is found in the context source and return its position and evaluation scores
def check_paper_in_context(query, data, relevant_paper):
    results = []
    paper_found = False
    for item in data:
        if item['query'] == query:
            other_papers_found = []
            for index, source in enumerate(item['context_source']):
                if relevant_paper in source['title']:
                    paper_found = True
                    query_generation_enabled = len(item['generated_queries']) > 1
                    results.append({
                        'query': query,
                        'paper_title': relevant_paper,
                        'position_with_paper': index + 1,
                        'answer_relevancy': item['evaluation_scores'].get('answer_relevancy', None),
                        'faithfulness': item['evaluation_scores'].get('faithfulness', None),
                        'contextual_relevancy': item['evaluation_scores'].get('contextual_relevancy', None),
                        'query_generation_enabled': query_generation_enabled,
                        'other_papers_found': 0,
                        'other_papers_list': ""
                    })
                else:
                    # Check if any other papers in the mapping appear in the context_source
                    for other_paper in paper_query_mapping.keys():
                        if other_paper in source['title'] and other_paper != relevant_paper:
                            other_papers_found.append(other_paper)

            if paper_found:
                results[-1]['other_papers_found'] = len(other_papers_found)
                results[-1]['other_papers_list'] = ", ".join(other_papers_found)
            else:
                query_generation_enabled = len(item['generated_queries']) > 1
                results.append({
                    'query': query,
                    'paper_title': relevant_paper,
                    'position_with_paper': None,
                    'answer_relevancy': item['evaluation_scores'].get('answer_relevancy', None),
                    'faithfulness': item['evaluation_scores'].get('faithfulness', None),
                    'contextual_relevancy': item['evaluation_scores'].get('contextual_relevancy', None),
                    'query_generation_enabled': query_generation_enabled,
                    'other_papers_found': len(other_papers_found),
                    'other_papers_list': ", ".join(other_papers_found)
                })

    return results

final_results = []

# Check each paper and its associated queries
for paper, queries in paper_query_mapping.items():
    for query in queries:
        results = check_paper_in_context(query, data, paper)
        final_results.extend(results)

# Convert the results to a DataFrame
df_results = pd.DataFrame(final_results)

# Save the results to a CSV file
output_file = os.path.join(current_dir, '..', 'data', 'random_papers_experiment3_results.csv')
df_results.to_csv(output_file, index=False)

print(f"Analysis results have been saved to {output_file}")