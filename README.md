## [2024-01] Efficient AI Final Project

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
To reproduce the magnitude pruning run below command: 

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
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored.

## Evaluation
```sh
python evaluate_llm.py --model_path MODEL_PATH
```

## Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) repository.
