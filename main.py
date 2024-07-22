import json
import os
from sklearn.model_selection import train_test_split

# Path to your .jsonl file
file_path = 'SalesKRA/Dataset.jsonl'

# Directory to save the individual JSON files
train_dir = 'train_data'
val_dir = 'val_data'

# Make sure the directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Read the .jsonl file and collect the entries
entries = []
with open(file_path, 'r') as file:
    for line in file:
        # Parse the JSON object from each line
        data = json.loads(line.strip())
        entries.append(data)

# Split the entries into train and validation sets
train_entries, val_entries = train_test_split(entries, test_size=0.2, random_state=42)

# Function to save a list of entries to a specified directory
def save_entries(entries, directory):
    for i, entry in enumerate(entries):
        # Construct a file name
        file_name = f"entry_{i+1}.json"
        # Construct the full path
        full_path = os.path.join(directory, file_name)
        # Write the entry to the file in JSON format
        with open(full_path, 'w') as f:
            json.dump(entry, f, indent=4)

# Save the train and validation entries to their respective directories
save_entries(train_entries, train_dir)
save_entries(val_entries, val_dir)

print(f"Saved {len(train_entries)} entries to {train_dir}")
print(f"Saved {len(val_entries)} entries to {val_dir}")
def formatting_func(example):
    text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
    return text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
   base_model_id,
   padding_side="left",
   add_eos_token=True,
   add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
   return tokenizer(formatting_func(prompt))

from datasets import load_dataset
​
# Define the function that will tokenize the data
def generate_and_tokenize_prompt(example):
   # Assuming you have a tokenizer loaded, e.g., tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   # Replace this with your actual tokenizer and tokenization logic
   example['input_ids'] = tokenizer.encode(example['text'], truncation=True, padding='max_length')
   return example
​
# Load the datasets from the JSON files in the directories
train_dataset = load_dataset('json', data_files={'train': f'{train_dir}/*.json'})
val_dataset = load_dataset('json', data_files={'validation': f'{val_dir}/*.json'})
​
# Tokenize the datasets
tokenized_train_dataset = train_dataset['train'].map(generate_and_tokenize_prompt)
tokenized_val_dataset = val_dataset['validation'].map(generate_and_tokenize_prompt)

max_length = 512 # This was an appropriate max length for my dataset
​
def generate_and_tokenize_prompt2(prompt):
   result = tokenizer(
       formatting_func(prompt),
       truncation=True,
       max_length=max_length,
       padding="max_length",
   )
   result["labels"] = result["input_ids"].copy()
   return result

def formatting_func(example):
   # Correct the keys according to your dataset format
   text = f"### Question: {example['text']}\n### Answer: {example['response']}"
   return text
​
def generate_and_tokenize_prompt2(example):
   # Tokenize the formatted text
   result = tokenizer(
       formatting_func(example),
       truncation=True,
       max_length=max_length,
       padding="max_length",
   )
   # Copy the input_ids to create labels for a language modeling task, if necessary
   result["labels"] = result["input_ids"].copy()
   return result
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = val_dataset.map(generate_and_tokenize_prompt2)
print(tokenized_train_dataset['train'][1]['input_ids'])

eval_prompt = " The following is KRA of Sales Executive: # "

tokenizer = AutoTokenizer.from_pretrained(
   base_model_id,
   add_bos_token=True,
)
​
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
​
model.eval()
with torch.no_grad():
   print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))
from peft import prepare_model_for_kbit_training
​
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
   """
   Prints the number of trainable parameters in the model.
   """
   trainable_params = 0
   all_param = 0
   for _, param in model.named_parameters():
       all_param += param.numel()
       if param.requires_grad:
           trainable_params += param.numel()
   print(
       f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
   )

print(model)

from peft import LoraConfig, get_peft_model
​
config = LoraConfig(
   r=32,
   lora_alpha=64,
   target_modules=[
       "q_proj",
       "k_proj",
       "v_proj",
       "o_proj",
       "gate_proj",
       "up_proj",
       "down_proj",
       "lm_head",
   ],
   bias="none",
   lora_dropout=0.05,  # Conventional
   task_type="CAUSAL_LM",
)
​
model = get_peft_model(model, config)
print_trainable_parameters(model)

print(model)
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
​
fsdp_plugin = FullyShardedDataParallelPlugin(
   state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
   optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
​
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

model = accelerator.prepare_model(model)
!pip install -q wandb -U
​
import wandb, os
wandb.login()
​
wandb_project = "journal-finetune"
if len(wandb_project) > 0:
   os.environ["WANDB_PROJECT"] = wandb_project

if torch.cuda.device_count() > 1: # If more than 1 GPU
   model.is_parallelizable = True
   model.model_parallel = True

import transformers
from datetime import datetime
​
project = "kra-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name
​
if tokenizer.pad_token is None:
   tokenizer.pad_token = tokenizer.eos_token
​
# You might need to resize the token embeddings in your model if you added a new token
model.resize_token_embeddings(len(tokenizer))
​
​
trainer = transformers.Trainer(
   model=model,
   train_dataset=tokenized_train_dataset['train'],
   eval_dataset=tokenized_val_dataset['validation'],
   args=transformers.TrainingArguments(
       output_dir=output_dir,
       warmup_steps=1,
       per_device_train_batch_size=2,
       gradient_accumulation_steps=1,
       max_steps=500,
       learning_rate=2.5e-5, # Want a small lr for finetuning
       bf16=True,
       optim="paged_adamw_8bit",
       logging_steps=25,              # When to start reporting loss
       logging_dir="./logs",        # Directory for storing logs
       save_strategy="steps",       # Save the model checkpoint every logging step
       save_steps=25,                # Save checkpoints every 50 steps
       evaluation_strategy="steps", # Evaluate the model every logging step
       eval_steps=25,               # Evaluate and save checkpoints every 50 steps
       do_eval=True,                # Perform evaluation at the end of training
       report_to="wandb",           # Comment this out if you don't want to use weights & baises
       run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
   ),
   data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
​
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

!huggingface-cli login --token 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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
​
base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)
​
base_model = AutoModelForCausalLM.from_pretrained(
   base_model_id,  # Mistral, same as before
   quantization_config=bnb_config,  # Same quantization config as before
   device_map="auto",
   trust_remote_code=True,
   use_auth_token=True
)
​
tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=Tru

from peft import PeftModel
​
ft_model = PeftModel.from_pretrained(base_model, "mistral-kra-finetune/checkpoint-300")

eval_prompt = " What are the tools in the market which helps sales person? # "
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
​
ft_model.eval()
with torch.no_grad():
   print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))
