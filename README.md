## [2024-01] Efficient AI Final Project

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
To reproduce the various pruning methods run below command: 

```sh
python main.py --model 42dot/42dot_LLM-PLM-1.3B --prune_method random --sparsity_ratio 0.5 --sparsity_type unstructured --save_model out/42_dot_weight_random
python main.py --model 42dot/42dot_LLM-PLM-1.3B --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save_model out/42_dot_weight_magnitude
python main.py --model 42dot/42dot_LLM-PLM-1.3B --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save_model out/42_dot_weight_sparsegpt_kobest --cali_data "kobest_hellaswag"
python main.py --model 42dot/42dot_LLM-PLM-1.3B --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save_model out/42_dot_weight_sparsegpt_wiki --cali_data "wikitext2"
python main.py --model 42dot/42dot_LLM-PLM-1.3B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save_model out/42_dot_wanda_kobest --cali_data "kobest_hellaswag"
python main.py --model 42dot/42dot_LLM-PLM-1.3B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save_model out/42_dot_wanda_wiki --cali_data "wikitext2"
```

We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`random`, `magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save_model`: Specifies the directory where the pruned model will be stored.
- `--cali_data`: Specifies the calibrating data for pruning with Wanda or SparseGPT, namely [`wikitext2`, `kobest_hellaswag`].

## Pruned Weights
We provide some pruned weights in this google drive [link](https://drive.google.com/drive/folders/17vAyTgZCKV9UPCFYuFI_NQA91k9_MLXH?usp=share_link).

## Evaluation with [kobest_helleaswag](https://www.google.com/search?client=safari&rls=en&q=kobest&ie=UTF-8&oe=UTF-8) Dataset and Check Sparsity
```sh
python evaluate_llm.py --model_path MODEL_PATH
```

## Acknowledgement
This repository is largely build upon the [Wanda](https://github.com/locuslab/wanda) repository.
