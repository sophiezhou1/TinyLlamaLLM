
# TinyLlama LLM

Framework for writing customer scopes of work  
Evonik SPE Regional Project 2025  
[Colab Notebook](https://colab.research.google.com/drive/1hp6l3ig7QBr9ygjRFpouf-GJdp_8Cii5?usp=sharing)
## Tools and Framework

#### Model and Fine Tuning
Base Model: [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)   
Fine Tuning: [QLoRA](https://arxiv.org/abs/2305.14314)

#### Training
HuggingFace Transformers: Model loading, tokenization, training  
PEFT: Integration of QLoRA  
bitsandbytes: 4-bit quantization  
accelerate

#### Data and Evaluation
Format: Instruction-Input-Response examples in Excel (saved as CSV)
- Instruction: Write the [cover letter] for a [feasibility study] given the following project description...
- Input: site information, project description, expectations, etc.
- Output: [Cover Letter]
## Integration

Load from HuggingFace (private repository)
```python
import huggingface_hub
huggingface_hub.login(token='')
```
Training

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instruction = examples["Instruction"]
    input_val = examples["Input"]
    response = examples["Response"]
    texts = []
    for instruction, input_val, response in zip(instruction, input_val, response):
        text = alpaca_prompt.format(instruction, input_val, response) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("sophiezhou1/study-feasibility", split="train")
print(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

```
## Dataset
[Instruction-Response (Excel)](https://evonik-my.sharepoint.com/:x:/p/s32717/Efdp-HJQwNVMoe7kpkOEcyEBxlnVnb6AegNbAQpZSXobXA?e=YGe49W)   
[HuggingFace: sophiezhou1/study-feasibility](https://huggingface.co/datasets/sophiezhou1/study-feasibility/settings)

If SharePoint link is no longer available, contact for access.
### Appendix

Code adapted from [Unsloth](https://docs.unsloth.ai/get-started/unsloth-notebooks).
### Feedback/Questions

If you have any questions, contact at zhou1278@purdue.edu.

