from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager
import sys

torch.manual_seed(0)
device = 'cuda'
# tokenizer = AutoTokenizer.from_pretrained("42dot/42dot_LLM-PLM-1.3B")
# model = AutoModelForCausalLM.from_pretrained("42dot/42dot_LLM-PLM-1.3B").to(device)
# tokenizer = AutoTokenizer.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_sparsegpt_wiki")
# model = AutoModelForCausalLM.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_sparsegpt_wiki").to(device)
# tokenizer = AutoTokenizer.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_magnitude")
# model = AutoModelForCausalLM.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_magnitude").to(device)
# tokenizer = AutoTokenizer.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_wanda_kobest_50")
# model = AutoModelForCausalLM.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_wanda_kobest_50").to(device)
tokenizer = AutoTokenizer.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_sparsegpt_kobest_50")
model = AutoModelForCausalLM.from_pretrained("/home/jocho/CourseWork/wanda/out/42_dot_weight_sparsegpt_kobest_50").to(device)
print(f"Model size: {model.get_memory_footprint():,} bytes")

lm = HFLM(pretrained=model, tokenizer=tokenizer, max_length=tokenizer.model_max_length,
              batch_size=8, trust_remote_code=True)

results = simple_evaluate(
            model=lm,
            tasks=["kobest_hellaswag"],
            task_manager=TaskManager(),)

print(make_table(results))