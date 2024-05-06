from os import pipe
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer

from config import FINETUNE_MODEL_DIR, DATASET, BASE_MODEL, OUT_MODEL, HUGGING_TOKKEN
from lora import BabCfg, LoraCfg
from hugging_signer import HuggingFaceNotebookLogin
from data_setter import DataSetter
from generate_promt import generate_prompt

def main():
  HuggingFaceNotebookLogin(HUGGING_TOKKEN).login()
  print("Logged in successfully!")
  data_set = DataSetter(DATASET).get_dataset()
  print("Loaded dataset successfully!")

  # Set up the model
  model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=BabCfg().get_config())
  tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
  tokenizer.padding_side = 'right'

  print("Set up the model successfully!")

  # Set up the training arguments
  trainer = SFTTrainer(
    model=model,
    train_dataset=data_set['train'],
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="outputs",
#        num_train_epochs = 1,
        max_steps=3000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        warmup_steps=0.03,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
    ),
    peft_config=LoraCfg().get_config(),
    formatting_func=generate_prompt,
  )


  # Train the model
  trainer.train()
  print("Trained the model successfully!")

  # Save the model
  ADAPTER_MODEL = "lora_adapter"
  trainer.model.save_pretrained(ADAPTER_MODEL)
  base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
  loar_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)
  finetune_model = loar_model.merge_and_unload()

  finetune_model.save_pretrained(OUT_MODEL)
  print("Saved the model successfully!")

main()