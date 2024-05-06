import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

class LoraCfg:
    def __init__(self):
        self.r = 6
        self.lora_alpha = 8
        self.lora_dropout = 0.05
        self.target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        self.task_type = "CAUSAL_LM"

    def get_config(self):
        return LoraConfig(r=self.r,lora_alpha=self.lora_alpha,lora_dropout=self.lora_dropout,target_modules=self.target_modules,task_type=self.task_type)
    
class BabCfg:
    def __init__(self):
        self.load_in_8bit = True
        self.bnb_8bit_quant_type = "nf4"
        self.bnb_8bit_compute_dtype = torch.float32

    def get_config(self):
        return BitsAndBytesConfig(load_in_8bit=self.load_in_8bit,bnb_8bit_quant_type=self.bnb_8bit_quant_type,bnb_8bit_compute_dtype=self.bnb_8bit_compute_dtype)
