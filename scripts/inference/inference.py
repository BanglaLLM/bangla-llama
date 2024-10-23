import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import logging
import re
import argparse
from datetime import datetime

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic LLM Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--question_list_path", type=str, required=True, help="Path to the CSV file containing questions")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV and logs")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for logging and results key")
    parser.add_argument("--retry_count", type=int, default=1, help="Number of retries for generating responses")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens for response generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature parameter for generation")
    return parser.parse_args()


# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    logging.info("Loading model and tokenizer from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logging.info("Model loaded successfully")
    return model, tokenizer

# Extract and pretty print the raw response gotten from the model
def extract_response(output):
    if isinstance(output, list):
        cleaned_output = ' '.join(output)
    else:
        cleaned_output = output
    
    # Remove special tokens
    cleaned_output = re.sub(r'<\|begin_of_text\|>|<\|end_of_text\|>', '', cleaned_output)
    
    # Extract text after "### Response:"
    response_match = re.search(r'### Response:(.*)', cleaned_output, re.DOTALL)
    if response_match:
        return response_match.group(1).strip()
    return cleaned_output.strip()


# generate response from the model according to the parameters define
def generate_response(model, tokenizer, question, max_new_tokens, temperature):
    alpaca_prompt = """
    এখানে একটি নির্দেশনা দেওয়া হলো, যা একটি কাজ সম্পন্ন করার উপায় বর্ণনা করে, এবং এর সাথে একটি ইনপুট দেওয়া হতে পারে যা আরও প্রেক্ষাপট প্রদান করে। একটি উত্তর লিখুন যা অনুরোধটি সঠিকভাবে পূরণ করে। \n 
    ### Instruction: {}
    ### Input: {}
    ### Response:"""
    ap = alpaca_prompt.format(question, "")
    inputs = tokenizer([ap], return_tensors="pt").to("cuda")
    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }
    output = model.generate(**inputs, **params)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]


# format the time 
def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def main():
    args = parse_args()

    # Setup logging
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime("%A_%B_%d_%Y_%I_%M_%p")
    
    # Define log file path
    log_directory = args.output_dir
    log_filename = f"inference_result_{args.model_name}_{formatted_time}.log"
    log_filepath = os.path.join(log_directory, log_filename)
    
    os.makedirs(log_directory, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_filepath,
        filemode='w'
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load question list
    questions_df = pd.read_csv(args.question_list_path)

    # Generate responses
    results = []
    for index, row in questions_df.iterrows():
        for ri in range(args.retry_count):
            question = row['Instruction']
            
            start_time = datetime.now()
            response_bangla = generate_response(model, tokenizer, question, args.max_new_tokens, args.temperature)

            end_time = datetime.now()
            gen_duration = end_time - start_time

            extracted_response_bangla = extract_response(response_bangla)
            
            q_id = f"{index}.{ri}"
            result = {
                'id': q_id,
                'question': question,
                args.model_name: extracted_response_bangla,
                'gen_duration': format_timedelta(gen_duration),
            }
            results.append(result)
            
            logging.info(f"Generating response..")
            logging.info("\n\n\n----------------------------------")
            logging.info(result)
            logging.info("----------------------------------\n\n\n")

    # Save results to CSV
    output_path = os.path.join(args.output_dir, f"inference_by_{args.model_name}_{formatted_time}.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"Responses saved to {output_path}")

if __name__ == "__main__":
    main()