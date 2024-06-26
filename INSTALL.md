# Installation  
Step 1: Create a new conda environment:
```
conda create -n effai python=3.9
conda activate effai
```
Step 2: Install relevant packages
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
```
There are known [issues](https://github.com/huggingface/transformers/issues/22222) with the transformers library on loading the LLaMA tokenizer correctly. Please follow the mentioned suggestions to resolve this issue.
Step 3: Install lm_eval packages following this [link](https://github.com/EleutherAI/lm-evaluation-harness)

or

Install via 
```
 conda env create --file environment.yaml
 conda activate effai
```
