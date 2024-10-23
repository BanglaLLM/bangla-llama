import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
from peft import PeftModel
import json

# Function to push model to Hugging Face hub
def push_to_hub(target_model_path, repo_id, hf_token):
    print("Pushing model to hub...")
    if os.path.exists(f"{target_model_path}/training_params.json"):
        training_params = json.load(open(f"{target_model_path}/training_params.json"))
        # Optionally, remove sensitive info if needed
        json.dump(training_params, open(f"{target_model_path}/training_params.json", "w"))

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=target_model_path, repo_id=repo_id, repo_type="model")

# Main function to handle the logic
def main(args):
    # Set up environment variables
    os.environ['PIP_CACHE_DIR'] = args.pip_cache_dir
    os.environ['HF_HOME'] = args.hf_home

    print(f"Hugging Face Home Directory: {os.environ['HF_HOME']}")
    print(f"Updated PATH: {os.environ['PATH']}")

    # Login to Hugging Face using the token via CLI
    os.system(f'huggingface-cli login --token {args.hf_token}')

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load adapter if applicable
    model_checkpoint = PeftModel.from_pretrained(model, args.adapter_path)

    # Merge the base model with adapters
    merged_model = model_checkpoint.merge_and_unload()

    # Fix any potential unbound error
    merged_model._hf_peft_config_loaded = False

    # Save merged model and tokenizer to target directory
    merged_model.save_pretrained(args.target_dir)
    tokenizer.save_pretrained(args.target_dir)

    # Push to Hugging Face hub
    push_to_hub(target_model_path=args.target_dir, repo_id=args.repo_id, hf_token=args.hf_token)

# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push fine-tuned model to Hugging Face Hub")
    
    parser.add_argument("--pip_cache_dir", type=str, default='/workspace/.pip/', help="Path to PIP cache directory")
    parser.add_argument("--hf_home", type=str, default='/workspace/.cache/huggingface/', help="Path to Hugging Face home directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the adapter directory containing 'adapter.safetensors'")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory to save the merged model and tokenizer")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face model repository ID")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face authentication token")
    
    args = parser.parse_args()
    main(args)
