import os
import torch
import json

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, logging
from peft import LoraConfig
from trl import SFTTrainer
import gc

from huggingface_hub import login
login()

import wandb
wandb.login()

wandb.init(
    # set the wandb project where this run will be logged
    project="Llama Guard Finetune"
)

torch.cuda.empty_cache()

model_id = "meta-llama/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16


competitor_data = load_dataset("json", data_files="./training_500_upd.json", split="train")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=torch.cuda.current_device(),
    quantization_config=quant_config,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="llama-guard-finetuned-3epochs-500",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch_fused",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    push_to_hub=True,                     # push model to hub
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=competitor_data,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()
trainer.save_model()
train_result = trainer.train()
print(train_result)