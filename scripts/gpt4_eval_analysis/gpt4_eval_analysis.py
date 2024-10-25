import pandas as pd
import argparse
import os

# Set up command-line arguments
parser = argparse.ArgumentParser(description="Merge CSV files and calculate average and min-max scores for model evaluations.")
parser.add_argument("--task_name", required=True, help="Task name for output file")
parser.add_argument("--input_file1", required=True, help="Path to the first input CSV file")
parser.add_argument("--input_file2", required=True, help="Path to the second input CSV file")
parser.add_argument("--output_path", required=True, help="Directory to save output files")
parser.add_argument("--model1_name", required=True, help="Name of the first model")
parser.add_argument("--model2_name", required=True, help="Name of the second model")

args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_path, exist_ok=True)

# Step 1: Read and merge CSV files
df1_i = pd.read_csv(args.input_file1)
df1 = df1_i[df1_i['Instruction'].notna()]
df2 = pd.read_csv(args.input_file2)
merged_df = pd.merge(df1, df2, left_on='Instruction', right_on='Prompt')
merged_df.drop(columns=['Prompt'], inplace=True)
merged_df.rename(columns={'Instruction': 'Prompt', 'coding': 'Category'}, inplace=True)
output_merged_file = os.path.join(args.output_path, f"merged_question_{args.task_name}.csv")
merged_df.to_csv(output_merged_file, index=False)
print("Merged CSV saved.")

# Step 2: Load merged CSV for score analysis
df = pd.read_csv(output_merged_file)
model1_name = args.model1_name
model2_name = args.model2_name

# Initialize dictionaries to store scores
category_scores_avg = {}
category_scores_minmax = {}

# Variables for tracking scores per instruction
prev_prompt = ""
prev_category = ""
scores_per_instruction_model1 = []
scores_per_instruction_model2 = []

for index, row in df.iterrows():
    if pd.isna(row["Prompt"]):
        continue

    category = row['Category'] if pd.notna(row['Category']) else "Others"
    score_model1 = row['Score_Model1']
    score_model2 = row['Score_Model2']

    if category not in category_scores_avg:
        category_scores_avg[category] = {'s_model1': [], 's_model2': []}

    category_scores_avg[category]['s_model1'].append(score_model1)
    category_scores_avg[category]['s_model2'].append(score_model2)

    prompt = row["Prompt"]

    if prev_prompt == prompt or prev_prompt == "":
        scores_per_instruction_model1.append(score_model1)
        scores_per_instruction_model2.append(score_model2)
    else:
        max_score_model1 = max(scores_per_instruction_model1)
        min_score_model2 = min(scores_per_instruction_model2)

        if prev_category not in category_scores_minmax:
            category_scores_minmax[prev_category] = {}

        category_scores_minmax[prev_category].setdefault('s_model1', []).append(max_score_model1)
        category_scores_minmax[prev_category].setdefault('s_model2', []).append(min_score_model2)

        scores_per_instruction_model1 = [score_model1]
        scores_per_instruction_model2 = [score_model2]

    prev_prompt = prompt
    prev_category = category

# Step 3: Calculate averages
scores_avg_simple = {
    category: {
        'model1': sum(scores['s_model1']) / len(scores['s_model1']),
        'model2': sum(scores['s_model2']) / len(scores['s_model2'])
    }
    for category, scores in category_scores_avg.items()
}

scores_avg_minmax = {
    category: {
        'model1': sum(scores['s_model1']) / len(scores['s_model1']),
        'model2': sum(scores['s_model2']) / len(scores['s_model2'])
    }
    for category, scores in category_scores_minmax.items()
}

results_avg_simple = pd.DataFrame(scores_avg_simple).T
results_avg_minmax = pd.DataFrame(scores_avg_minmax).T

# Rename columns and save results to CSV and Excel
results_avg_simple.columns = results_avg_simple.columns.fillna('')
results_avg_minmax.columns = results_avg_minmax.columns.fillna('')
results_avg_simple.rename(columns={'model1': model1_name, 'model2': model2_name}, inplace=True)
results_avg_minmax.rename(columns={'model1': model1_name, 'model2': model2_name}, inplace=True)

results_avg_simple.to_csv(os.path.join(args.output_path, f"{args.task_name}_average_scores_simple.csv"))
results_avg_minmax.to_csv(os.path.join(args.output_path, f"{args.task_name}_average_scores_minmax.csv"))

print("Score analysis results saved.")
