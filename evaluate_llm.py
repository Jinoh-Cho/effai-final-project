from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager
from lib.prune import check_sparsity
import argparse
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to load the pruned/original model.')
    args = parser.parse_args()

    device = 'cuda:0'
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    print(f"Model size: {model.get_memory_footprint():,} bytes")

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"Sparsity Check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################

    lm = HFLM(pretrained=model, tokenizer=tokenizer, max_length=tokenizer.model_max_length,
                batch_size=8, trust_remote_code=True)

    results = simple_evaluate(
                model=lm,
                tasks=["kobest_hellaswag"],
                task_manager=TaskManager(),)

    print(make_table(results))