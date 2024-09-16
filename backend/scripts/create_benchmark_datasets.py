import pandas as pd
import os
import shutil
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Add the PYTHONPATH from the .env file
sys.path.append(os.getenv("PYTHONPATH"))

# Read the relevant papers
relevant_papers = pd.read_csv('data/benchmark/title_abstracts.csv')

# Read the irrelevant papers
irrelevant_papers = pd.read_csv('data/benchmark/irrelevant_papers.csv')


# Iterate through each row of relevant_papers.xlsx
for index, row in relevant_papers.iterrows():
    # Copy the "Title" and "Abstract"
    title = row['Title']
    abstract = row['Abstract']
    
    # Create a clone of irrelevant_papers.csv
    new_filename = os.path.join('data/benchmark', f'benchmark_dataset_{index + 1}.csv')
    shutil.copy('data/benchmark/irrelevant_papers.csv', new_filename)
    
    # Read the newly created file
    benchmark_df = pd.read_csv(new_filename)
    
    # Append the new row with "Title" and "Abstract"
    new_row = pd.DataFrame({'Title': [title], 'Abstract': [abstract]})
    benchmark_df = pd.concat([benchmark_df, new_row], ignore_index=True)
    
    # Save the updated DataFrame
    benchmark_df.to_csv(new_filename, index=False)

    # Break the loop if we've created 16 files
    if index == 19:  # 0-based index, so 19 is the 20th row
        break

print(f"Created {index + 1} benchmark dataset files in data/benchmark_2")