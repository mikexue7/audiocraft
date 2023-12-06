from pathlib import Path
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenForCausalLM
from peft import LoraModel, LoraConfig, get_peft_model
import pdb

model_name = "musicgen-small"
model = MusicgenForCausalLM.from_pretrained("facebook/musicgen-small")

pdb.set_trace()

config = LoraConfig(r=16, target_modules=['k_proj', 'q_proj', 'v_proj', 'out_proj'], lora_alpha=32, lora_dropout=0.01)

lora_model = get_peft_model(model, config)

print("Converted model to LoRA")

root = Path.home() / 'audiocraft' / 'checkpoints'

torch.save({'best_state': {'model': lora_model.state_dict()}}, root / f'lora_{model_name}.th')

print("Saved checkpoint!")