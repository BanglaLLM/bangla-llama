**\`Training details:**  
We used the Low-Rank Adaptation (LoRA) technique, which optimizes computational resources while ensuring the model's robustness and generalization while retaining good performance by carefully updating a small subset of model parameters.

NVIDIA A100 GPUs, known for their high-performance processing capabilities ideal for deep learning applications, were utilized for all experiments.

**Pretraining phase:**

For pretraining, we utilized the "uonlp/CulturaX" dataset, specifically the Bengali (bn) subset with **12.4M rows**. This dataset was chosen for its comprehensive coverage of Bengali language content. 

Model Architecture and Training Configuration: We employed the Hugginface Transformer’s LlamaForCausalLM architecture, which is well-suited for generative language tasks. The tokenizer used was AutoTokenizer, ensuring compatibility with the LLaMA model's vocabulary. To maintain high precision during training, we opted not to use 8-bit or 4-bit quantization, instead utilizing **full precision parameters**.(add the tokenizer of banglallama 2 that is being developed by us)

The model was trained with a sequence length of 4096 tokens, allowing it to capture long-range dependencies effectively. We implemented sample packing to optimize GPU memory usage and increase training efficiency. The pad\_to\_sequence\_len option was enabled to ensure consistent input sizes.

Training Process: We utilized the LoRA (Low-Rank Adaptation) method for efficient fine-tuning. The LoRA configuration included a rank (r) of 32, an alpha of 16, and a dropout rate of 0.05. We targeted all linear layers for adaptation and included both the embedding layer and the language model head in the modules to save.

The training process was set for a maximum of 10,000 steps, with a learning rate of 0.0002 using the **AdamW optimizer with 8-bit precision**. We employed a cosine learning rate scheduler with a warmup period of 10 steps. Gradient accumulation was set to 8 steps with a micro batch size of 1, effectively creating a larger batch size to stabilize training.

To further optimize training, we enabled gradient checkpointing and flash attention, leveraging the capabilities of modern GPU architectures. The model was trained in mixed precision, utilizing **bfloat16** where supported by the hardware.

Training Duration: The model was trained for one epoch over the entire dataset. The 8B model required  **41 hours and 21 Minutes** of training, while the 1B model required **10 hours and 6 minutes** of training on Runpod’s A100SXM/PCI instance with 80GB VRAM.(can add some context about llama 2\)

**Pretraining Parameters**:

| Configurations | BongLLama 2  | BongLLama 3 |
| :---- | :---- | :---- |
| Training Steps |  | 10000 |
| Epochs |  | 1 |
| Micro Batch Size |  | 1 |
| Gradient Accumulation Steps |  | 8 |
| Initial Learning Rate |  | 2e-4 |
| LoRA Rank (r) |  | 32 |
| LoRA Alpha |  | 16 |
| LoRA Dropout |  | 0.05 |
| Training Precision |  | bf16 (auto) |
| Max Sequence Length |  | 4096 |
| Optimizer |  | AdamW (8-bit) |
| Learning Rate Scheduler |  | Cosine |
| Warmup Steps |  | 10 |
| Gradient Checkpointing |  | Enabled |
| Flash Attention |  | Enabled |
| Sample Packing |  | Enabled |
| LoRA Target Modules |  | All linear layers |
|  |  |  |
|  |  |  |

**Finetunning phase:**

Following the pretraining phase, our BongLLaMA model’s underwent a fine-tuning process to enhance its performance on specific Bengali language tasks. This fine-tuning stage was crucial for aligning the model with Bengali-specific instructions and improving its ability to understand and generate Bengali text in various contexts.

Base Model and Dataset: We initiated the fine-tuning process using our pretrained models,, which had already been exposed to a large corpus of Bengali text during the pretraining phase. For fine-tuning, we employed the **BanglaLLM/bangla-alpaca-orca dataset,** a comprehensive collection of **172,000 Bengali instructions**. This dataset was specifically curated to include a wide range of **Bengali-centric tasks, cultural nuances, and linguistic patterns.**

Model Architecture and Training Configuration: **We continued to use the LlamaForCausalLM architecture, maintaining consistency with the pretraining phase**. The AutoTokenizer was employed to ensure proper tokenization of Bengali text. To preserve the model's full capabilities, we opted against 8-bit or 4-bit quantization, instead utilizing full precision parameters.

The fine-tuning process maintained a sequence length of 4096 tokens, allowing the model to handle complex, long-form Bengali instructions and responses..

LoRA Implementation: We persisted with the Low-Rank Adaptation (LoRA) methodology for fine-tuning, which allows for efficient parameter updates while preserving the base model's general knowledge. The LoRA configuration remained consistent with the pretraining phase, using a rank (r) of 32, an alpha of 16, and a dropout rate of 0.05. We targeted all linear layers for adaptation and included both the embedding layer and the language model head in the modules to save.

Training Process: The fine-tuning was conducted for **one epoch over the entire dataset of 172,000 instructions**. We used a **learning rate of 0.0002 with the AdamW optimizer (8-bit precision) and a cosine learning rate scheduler**. The **warmup period was set to 10** steps. We maintained a gradient accumulation of 8 steps with a micro batch size of 1, effectively creating a larger batch size of 8 to balance between training stability and memory constraints.

To optimize the fine-tuning process, we continued to use gradient checkpointing and flash attention, leveraging the capabilities of our GPU architecture. The model was trained in mixed precision, utilizing bfloat16 where supported by the hardware.

Evaluation and Checkpointing: During fine-tuning, we allocated 5% of the dataset for validation (val\_set\_size: 0.05). Evaluations were conducted four times per epoch. 

Hardware and Infrastructure: The fine-tuning process was carried out on NVIDIA A100 GPUs, providing the high-performance computing capabilities necessary for efficient training of large language models. 

Outcome: This fine-tuning approach aimed to enhance our BongLLaMA model's proficiency in understanding and generating Bengali text, with a particular focus on instruction-following capabilities. The resulting model, tailored to Bengali language nuances and cultural contexts, represents a significant step towards creating a high-quality, Bengali-specific large language model.(**Optional I think)**

| Configurations | BongLLama 2  | BongLLama 3 |
| :---- | :---- | :---- |
| Training Data |  | 172k instructions (bangla-alpaca-orca) |
| Epochs |  | 1 |
| Effective Batch Size |  | 8  |
| Gradient Accumulation Steps |  | 8 |
| Initial Learning Rate |  | 2e-4 |
| LoRA Rank (r) |  | 32 |
| LoRA Alpha |  | 16 |
| LoRA Dropout |  | 0.05 |
| Training Precision |  | bf16 (auto) |
| Max Sequence Length |  | 4096 |
| Optimizer |  | AdamW (8-bit) |
| Learning Rate Scheduler |  | Cosine |
| Warmup Steps |  | 10 |
| Gradient Checkpointing |  | Enabled |
| Flash Attention |  | Enabled |
| Sample Packing |  | Enabled |
| LoRA Target Modules |  | All linear layers |
|  |  |  |
|  |  |  |

# Generation 

Assesment Approach:

To assess the performance of our BongLLaMA models, we implemented a comprehensive evaluation strategy that combines automated scoring. This approach allows us to efficiently evaluate a large number of responses while ensuring the accuracy and reliability of the scores.

Evaluation Methodology:

1. GPT-4 Omni Scoring: We leveraged the capabilities of OpenAI's latest GPT-4 Omni model for our primary scoring mechanism. GPT-4 Omni represents the cutting edge in AI language models, offering enhanced versatility, depth, and understanding of complex human interactions. Its advanced capabilities in multitasking and real-time problem-solving make it an ideal tool for evaluating the responses of our BongLLaMA models. For each query-response pair, GPT-4 Omni was tasked with assigning a score on a 100-point scale. This scale was chosen to provide a fine-grained assessment of response quality, allowing for nuanced differentiation between model performances.

Evaluation Suite:

Our assessment suite was carefully designed to provide a comprehensive evaluation of BongLLaMA's capabilities across various domains. The suite includes over 120 diverse examples, covering a wide range of topics and task types, such as:

* Question Answering  
* Reasoning  
* **Bengali Literature and Culture(literature can be called by this)**  
* Entertainment  
* Translation  
* Programming  
* Ethics and Social Understanding(can call **open\_qa**)  
* General Knowledge

Scoring and Analysis:

For each task category, we computed an overall score by summing the individual sample scores and normalizing to a 100-point scale. This approach provides a holistic view of our models' performance across different domains, allowing us to:

* Identify areas of strength and weakness  
* Compare performance across different versions of our models  
* Benchmark our models against other existing Bengali language models or multilingual models with Bengali capabilities

(Optional I think)By combining the efficiency of GPT-4 Omni's automated scoring with the nuanced understanding provided by human experts, our evaluation process offers a robust and comprehensive assessment of BongLLaMA's capabilities. This methodology not only provides valuable insights into our models' performance but also sets a standard for evaluating large language models in Bengali, contributing to the broader field of NLP for low-resource languages.

# 

# Generation Parameters:

The quality of text generation in large language models is significantly influenced by the choice of generation parameters during inference. For our BongLLaMA models, we carefully selected these parameters to balance coherence, relevance, and creativity in the generated responses. Below, we detail our generation setup and the specific parameters used:

Hardware Configuration: All generation tasks were performed on an **NVIDIA RTX A6000 GPU with 48GB of VRAM**. This high-performance hardware allowed us to run our models efficiently and handle complex generation tasks without memory constraints.

Generation Parameters: We fine-tuned our generation process with the following parameters:

1. Model Loading Configuration: We loaded the model in full precision using the bfloat16 data type, balancing high computational capabilities with memory efficiency and numerical stability.  
2. Temperature: We set the **temperature to 0.6**. This value strikes a balance between deterministic outputs and creative diversity. It allows our model to produce responses that are coherent and relevant while still maintaining a degree of variability and unexpectedness in the generated text.  
3. Maximum New Tokens: The maximum number of new tokens for each generation was set to **2048**. This generous limit allows the model to produce comprehensive and detailed responses when necessary, without unnecessarily truncating complex explanations or long-form content.  
4. Context Size: We maintained the model's default **context size of 4096 tokens for llama 2 and 8192 for llama 3**, allowing it to consider a substantial amount of preceding context when generating responses.

**Multiple Generations**: To account for the inherent variability in language model outputs and to ensure a more robust evaluation, we generated responses to each question three times. This approach allows us to:

1. Assess the consistency of our model's performance across multiple generations.  
2. Identify any potential instabilities or inconsistencies in the model's outputs.  
3. Provide a more representative sample of the model's capabilities for each question.

Other Considerations:(Probably can add into appendix or in the main part if can)

* **We did not implement top-k or top-p sampling** in our generation process, allowing the model to sample from its full vocabulary distribution as guided by the temperature setting.  
* **No specific repetition penalty** was applied, relying instead on the model's inherent learning from the fine-tuning process to avoid redundant text.  
* We did not apply additional quantization techniques during the generation phase, utilizing the model in its full precision to maintain optimal performance.

Evaluation Process: For each query in our evaluation suite, we used these generation parameters consistently across all versions of our BongLLaMA models, generating three responses for each question. This approach ensured a fair comparison between different model iterations and allowed us to assess the true capabilities and consistency of each model version under identical generation conditions.

**Safi : Shorter Version, Zehadi to Finalise** 

**Longer version can be added to appendix** 

### **Training Details**

BongLLaMA models were trained using the Low-Rank Adaptation (LoRA) technique to optimize computational efficiency while maintaining performance. NVIDIA A100 GPUs were used, and the pretraining utilized the "uonlp/CulturaX" Bengali dataset (12.4M rows). The model employed Huggingface’s LlamaForCausalLM architecture, with AutoTokenizer ensuring compatibility with the LLaMA model's vocabulary. Training was conducted with full precision, and no quantization was applied. A sequence length of 4096 tokens was used, with sample packing enabled to optimize memory usage.

### **Pretraining Configuration**

Key configurations included a LoRA rank of 32, alpha of 16, dropout of 0.05, a learning rate of 0.0002, and AdamW as the optimizer. Gradient checkpointing and flash attention were utilized, and the training was set for 10,000 steps. The 8B model required 41 hours and 21 minutes of training, while the 1B model took 10 hours and 6 minutes.

### **Fine-Tuning Phase**

The fine-tuning phase aimed to align the model with specific Bengali tasks using the BanglaLLM/bangla-alpaca-orca dataset, consisting of 172,000 Bengali instructions. The architecture and LoRA configuration were consistent with pretraining. One epoch of fine-tuning was conducted using a learning rate of 0.0002, with an effective batch size of 8 and gradient accumulation of 8 steps. The model's proficiency was further enhanced to handle Bengali-specific instructions.

### **Generation Parameters**

During text generation, a temperature of 0.6 was applied to strike a balance between deterministic outputs and creative diversity. This allowed the model to generate responses that were coherent yet varied, ensuring a mix of consistency and creativity in the output. Additionally, the maximum number of new tokens was set to 2048, with a context size of 4096 tokens for LLaMA 2 models and 8192 tokens for LLaMA 3 models.

### **Evaluation**

The performance of BongLLaMA models was assessed using GPT-4 Omni, scoring responses on a 100-point scale. Evaluation covered over 120 tasks across domains like question answering, reasoning, literature, ethics, and programming. This evaluation helped identify areas of strength and weakness, with scores normalized for comparative analysis.

