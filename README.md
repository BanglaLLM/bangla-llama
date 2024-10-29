# Bangla-Llama: A Family of LLaMA-based LLMs focused on Bangla Language

## Description

This repository contains the code and models for "Bangla-Llama", a project focused on enhancing the performance of language models for the Bangla language. It builds upon the open-source LLaMA model, introducing additional Bangla tokens and employing the LoRA methodology for efficient training. Please read the technical report for more details.


# Model Management and Evaluation Scripts

This repository contains a set of scripts designed for managing, evaluating, and deploying machine learning models, specifically focusing on language models. The scripts are organized to handle different tasks such as inference, evaluation, and pushing models to the Hugging Face Hub.

## Table of Contents

- [Scripts Overview](#scripts-overview)
  - [Inference Script](#inference-script)
  - [Evaluation Analysis Script](#evaluation-analysis-script)
  - [Model Push Script](#model-push-script)
- [Usage](#usage)
  - [Inference Script Usage](#inference-script-usage)
  - [Evaluation Analysis Script Usage](#evaluation-analysis-script-usage)
  - [Model Push Script Usage](#model-push-script-usage)
- [Requirements](#requirements)
- [Setup](#setup)

## Scripts Overview

### Inference Script

**File:** `scripts/inference/inference.py`

The inference script is designed to generate responses from a language model based on a list of questions provided in a CSV file. It supports dynamic loading of models and tokenizers, and logs the inference process.

**Key Features:**
- Loads a pre-trained language model and tokenizer.
- Generates responses for each question in the input CSV.
- Supports retry logic for generating responses.
- Logs the inference process and saves results to a CSV file.

### Evaluation Analysis Script

**File:** `scripts/gpt4_eval_analysis/gpt4_eval_analysis.py`

This script merges two CSV files containing model evaluation results and calculates average and min-max scores for model evaluations. It is useful for analyzing the performance of different models on a set of tasks.

**Key Features:**
- Merges evaluation results from two CSV files.
- Calculates average and min-max scores for each category.
- Saves the analysis results to CSV files.

### Model Push Script

**File:** `scripts/push_model/push.py`

The model push script is used to upload a fine-tuned model to the Hugging Face Hub. It handles the merging of model adapters and ensures the model is properly configured before uploading.

**Key Features:**
- Loads and merges model adapters.
- Saves the merged model and tokenizer.
- Pushes the model to a specified Hugging Face repository.

### Evaluation Script

**File:** `scripts/gpt4_eval/eval.py`

This script is used to evaluate the responses of two large language models (LLMs) for the same instruction. It scores the responses on a scale of 1-100 and provides detailed reasoning for the scores. The script rotates through a list of models to avoid rate limits and saves the results in a CSV file.

**Key Features:**
- Evaluates and scores model responses.
- Rotates through a list of models to handle rate limits.
- Saves results to a CSV file and a pickle backup.

## Usage

### Evaluation Script Usage

To run the evaluation script, use the following command:

```bash
python scripts/gpt4_eval/eval.py --model_type <MODEL_TYPE> --model_name <MODEL_NAME> --output_path <OUTPUT_PATH> --output_file <OUTPUT_FILE> --input_file1 <CSV1_PATH> --input_file2 <CSV2_PATH> --sleep_time <SLEEP_TIME> --sleep_type <SLEEP_TYPE> --eval_name <EVAL_NAME>
```

- `--model_type`: Type of model to use (e.g., "openai", "anthropic", "llama").
- `--model_name`: Name of the model to use.
- `--output_path`: Directory to save output files.
- `--output_file`: Path to save the output CSV file.
- `--input_file1`: Path to the first input CSV file.
- `--input_file2`: Path to the second input CSV file.
- `--sleep_time`: Time to sleep between model rotations.
- `--sleep_type`: Type of sleep interval.
- `--eval_name`: Evaluation name.

### Inference Script Usage

To run the inference script, use the following command:

```bash
python scripts/inference/inference.py --model_path <MODEL_PATH> --question_list_path <CSV_PATH> --output_dir <OUTPUT_DIR> --model_name <MODEL_NAME> [--retry_count <RETRY_COUNT>] [--max_new_tokens <MAX_TOKENS>] [--temperature <TEMPERATURE>]
```

- `--model_path`: Path to the pre-trained model.
- `--question_list_path`: Path to the CSV file containing questions.
- `--output_dir`: Directory to save the output CSV and logs.
- `--model_name`: Name of the model for logging and results key.
- `--retry_count`: Number of retries for generating responses (default: 1).
- `--max_new_tokens`: Maximum number of new tokens for response generation (default: 2048).
- `--temperature`: Temperature parameter for generation (default: 0.6).

### Evaluation Analysis Script Usage

To run the evaluation analysis script, use the following command:
```bash
python scripts/gpt4_eval_analysis/gpt4_eval_analysis.py --task_name <TASK_NAME> --input_file1 <CSV1_PATH> --input_file2 <CSV2_PATH> --output_path <OUTPUT_PATH> --model1_name <MODEL1_NAME> --model2_name <MODEL2_NAME>
```


- `--task_name`: Task name for the output file.
- `--input_file1`: Path to the first input CSV file.
- `--input_file2`: Path to the second input CSV file.
- `--output_path`: Directory to save output files.
- `--model1_name`: Name of the first model.
- `--model2_name`: Name of the second model.

### Model Push Script Usage

To run the model push script, use the following command:
```bash
python scripts/push_model/push.py --pip_cache_dir <PIP_CACHE_DIR> --hf_home <HF_HOME> --model_path <MODEL_PATH> --adapter_path <ADAPTER_PATH> --target_dir <TARGET_DIR> --repo_id <REPO_ID> --hf_token <HF_TOKEN>
```

- `--pip_cache_dir`: Path to PIP cache directory (default: `/workspace/.pip/`).
- `--hf_home`: Path to Hugging Face home directory (default: `/workspace/.cache/huggingface/`).
- `--model_path`: Path to the fine-tuned model.
- `--adapter_path`: Path to the adapter directory containing `adapter.safetensors`.
- `--target_dir`: Directory to save the merged model and tokenizer.
- `--repo_id`: Hugging Face model repository ID.
- `--hf_token`: Hugging Face authentication token.

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers library
- Hugging Face Hub
- Pandas
- Other dependencies as specified in the scripts

## Setup

1. Clone the repository.
2. Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your environment variables and paths as needed.
4. Run the scripts using the provided usage instructions.



## Contact

For any queries regarding the codebase or research, please reach out to Abdullah Khan Zehady at brishtiteveja@gmail.com.
