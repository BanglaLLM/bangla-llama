from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain_community.llms import OpenAIChat
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence

from dotenv import load_dotenv
import pandas as pd
import json
import os
import time
import pickle
import re
import argparse

# Load environment variables
load_dotenv()
# SLEEP_TIME= 1800


# SLEEP_TYPE = 30

MODEL_LIST = [
    {'name': 'gpt-4o', 'type': 'openai'},
    {'name': 'gpt-4-0613', 'type': 'openai'},
    {'name': 'gpt-4', 'type': 'openai'},
    {'name': 'gpt-3.5-turbo', 'type': 'openai'},
    {'name': 'claude-2', 'type': 'anthropic'},
    {'name': 'llama-3-70b', 'type': 'llama', 'api_url': 'http://your-llama-api-url'}
]

current_model_index = 0


# define API keys in the .env file
API_KEYS = {
    'openai': os.getenv("OPENAI_API_KEY"),
    'anthropic': os.getenv("ANTHROPIC_API_KEY"),
    'llama': os.getenv("LLAMA_API_KEY")
}

system_message = """You are a professional AI data evaluation researcher who possesses deep knowledge on AI model architecture and responses. You will help in model evaluation."""

human_message_template = """Task: Our goal is to evaluate generated response of two Large Language models for the same instruction. Can you score this pair of response between 1-100 and give your reasoning behind score in full details. Please produce your response in valid json format. Keep your reasoning as long as possible. Keep bulletpointed detail, deep reasoning.

Question: {question}

Model1 Response:
{resp_1}

Model2 Response:
{resp_2}

Please provide your evaluation in the following JSON format:
{{
  "Model1": {{
    "Score": <score between 1-100>,
    "Reasoning": [
      <list of detailed bullet points explaining the score>
    ]
  }},
  "Model2": {{
    "Score": <score between 1-100>,
    "Reasoning": [
      <list of detailed bullet points explaining the score>
    ]
  }}
}}

Ensure your response is valid JSON.
"""


def prepare_chain_with_model(model_name, model_type, api_url=None):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message_template)
    ])
    
    if model_type == 'openai':
        chat = ChatOpenAI(model_name=model_name, api_key=API_KEYS['openai'])
    elif model_type == 'anthropic':
        chat = ChatAnthropic(model=model_name, api_key=API_KEYS['anthropic'])
    elif model_type == 'llama':
        # Assuming Llama model can be used with OpenAIChat
        chat = OpenAIChat(model_name=model_name, api_base=api_url, api_key=API_KEYS['llama'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    chain = RunnableSequence(chat_prompt | chat)
    return chain 

def extract_json_content(response):
    match = re.search(r'\{.*\}', response, re.DOTALL)
    return match.group(0) if match else None

def rotate_model():
    global current_model_index
    current_model_index = (current_model_index + 1) % len(MODEL_LIST)
    return MODEL_LIST[current_model_index]

def generate_and_save(file1_path, file2_path, output_path, output_file, model_type, model_name, test_rows=None, no_skip=True):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    
    #merged_df = pd.merge(df1, df2, on='question', how="inner", suffixes=('_1', '_2'))

    # Create a dictionary from df2 for faster lookup
    df2_dict = dict(zip(df2['question'], df2['metallama']))

    # Manually merge the dataframes
    merged_data = []
    for _, row in df1.iterrows():
        question = row['question']
        if question in df2_dict:
            merged_data.append({
                'question': question,
                'banglallama': row['banglallama'],
                'metallama': df2_dict[question]
            })

    merged_df = pd.DataFrame(merged_data)
    print(f"Shape of manually merged df: {merged_df.shape}")

    if test_rows:
        merged_df = merged_df.head(test_rows)

    pickle_df_path = output_path + "/pickle_backup_dfs"
    pickle_path = f"{os.path.splitext(pickle_df_path)[0]}/pickle_{model_type}_{model_name}_{EVAL_NAME}.pkl"
    
    # Load existing results if available
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            result_df = pickle.load(f)
        print(f"Loaded {len(result_df)} existing results from {pickle_path}")
    else:
        result_df = pd.DataFrame()

    current_model = next((m for m in MODEL_LIST if m['name'] == model_name and m['type'] == model_type), None)
    if not current_model:
        raise ValueError(f"Model {model_name} of type {model_type} not found in MODEL_LIST")

    for index, row in merged_df.iterrows():
        question = row['question']

        # Skip if this question has already been processed, unless no_skip is True
        if not no_skip and not result_df.empty and question in result_df['Prompt'].values:
            print(f"Skipping already processed question: {question}")
            continue

        resp_1 = row['banglallama']
        resp_2 = row['metallama']

        success = False
        while not success:
            try:
                chain = prepare_chain_with_model(current_model['name'], current_model['type'], current_model.get('api_url'))

                response = chain.invoke({"question": question, "resp_1": resp_1, "resp_2": resp_2})

                # Check if 'content' is an attribute of the response
                if hasattr(response, 'content'):
                    response_content = response.content
                elif isinstance(response, str):
                    response_content = response
                else:
                    print("Unexpected response format:")
                    print(response)
                    raise ValueError("Unable to extract content from response")

                json_string = extract_json_content(response_content)

                if json_string:
                    comparison = json.loads(json_string)
                    print("Successfully parsed JSON:")
                    print(json.dumps(comparison, indent=2))

                    banglallama_score_1 = comparison['Model1']['Score']
                    metallama_score = comparison['Model2']['Score']
                    model1_reasoning = " ".join(comparison['Model1']['Reasoning'])
                    model2_reasoning = " ".join(comparison['Model2']['Reasoning'])

                    new_row = pd.DataFrame({
                        'Prompt': [question],
                        'Resp_Model1': [resp_1],
                        'Resp_Model2': [resp_2],
                        'Score_Model1': [f"{banglallama_score_1}"],
                        'Score_Model2': [f"{metallama_score}"],
                        'Reasoning_Model1': [model1_reasoning],
                        'Reasoning_Model2': [model2_reasoning],
                        'Evaluator_Model': [f"{current_model['type']}-{current_model['name']}"]
                    })

                    result_df = pd.concat([result_df, new_row], ignore_index=True)

                    # Save to pickle after each successful response
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(result_df, f)

                    print(f"Updated results saved to {pickle_path}")
                    success = True
                else:
                    print("No JSON-like content found in the response.")
                    raise Exception("Invalid response format")

            except Exception as e:
                print(f"Error: {e}")
                current_model = rotate_model()
                print(f"Rotated to model: {current_model['name']} ({current_model['type']})")

                if current_model_index == 0:
                    print("All models exhausted. Waiting for 30 minutes due to rate limit...")
                    time.sleep(SLEEP_TIME)

    # Save final result to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save model comparisons.")
    parser.add_argument("--model_type", default="openai", choices=["openai", "anthropic", "llama"], help="Type of model to use")
    parser.add_argument("--model_name", default="gpt-4o", help="Name of the model to use")
    parser.add_argument("--output_path", default=f"/workspace/bangla-llama/evaluation/gpt4_eval_output", help="Output file path")
    parser.add_argument("--output_file", default=f"/workspace/bangla-llama/evaluation/gpt4_eval_output/{EVAL_NAME}.csv", help="Output file path")
    parser.add_argument("--input_file1", default="/workspace/bangla-llama/evaluation/output/inference_result_banglallama-3-8b_instruct_Saturday_October_05_2024_02_42_PM.csv", help="First input file path")
    parser.add_argument("--input_file2", default="/workspace/bangla-llama/evaluation/output/inference_by_metallama-3-8b_Sunday_October_06_2024_05_07_PM.csv", help="Second input file path")
    parser.add_argument("--test_rows", type=int, help="Number of rows to process for testing")
    parser.add_argument("--no_skip", action="store_false", help="Do not skip already processed questions")
    parser.add_argument("--sleep_time", type=int, default=1800, help="Time to sleep between model rotations")
    parser.add_argument("--sleep_type", type=int, default=30, help="Type of sleep interval")
    parser.add_argument("--eval_name", default='BnLLama3.1_8b_ins_vs_MetaLLama3.1_8b_base', help="Evaluation name")

    args = parser.parse_args()

    # Use the command-line arguments
    SLEEP_TIME = args.sleep_time
    SLEEP_TYPE = args.sleep_type
    EVAL_NAME = args.eval_name

    generate_and_save(args.input_file1, args.input_file2, args.output_path, args.output_file, args.model_type, args.model_name, args.test_rows, args.no_skip)