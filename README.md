# Llama2-LoRA-Trainer

## Introduction
This is the official impletation repository of NAACL findings paper, [GenTKG: Generative Forecasting on Temporal Knowledge Graph](https://arxiv.org/abs/2310.07793)

This work is about fine-tuning the large language model llama2-7B with [peft](https://github.com/huggingface/peft) and using it for temporal knowledge graph (tkg) forecasting. The training and evaluation data used are obtained by TLR retreival, and the FIT trained model weights are stored on [Google Drive](https://drive.google.com/drive/folders/1pZwppUnjLAfdzu30lKaxZGM3AC-x5We-?usp=drive_link).

## Setup
### Environment

Download the codes and go to the folder:

```javascript
git clone https://github.com/mayhugotong/TKGForcaster.git
cd TKGForcaster
```
Create an environment:
```
conda create -n gtkg python=3.8
conda activate gtkg
pip install -r requirements.txt 
pip install git+https://github.com/huggingface/peft.git
```
Download data and models from [Google Drive](https://drive.google.com/drive/folders/1pZwppUnjLAfdzu30lKaxZGM3AC-x5We-?usp=drive_link) and then unzip and save them in folders "data" and "model".

You can use gdown to do it:
```
pip install gdown
gdown https://drive.google.com/file/d/1C63Ugg_Xc1MGgeToiYNM0X4i35CJUEWA/view?usp=sharing
unzip data.zip -d .
gdown https://drive.google.com/file/d/1pZcUDot9kmcjP2a4d10QVo86WGl3Bj9t/view?usp=sharing
unzip model.zip -d .
gdown https://drive.google.com/file/d/145avybZXtlTrshVBJ22B6KSJPnK5nQVS/view?usp=sharing
unzip model_backup.zip -d .
```

### Lexical datasets
Before anything, you might want to create datasets in lexicons instead of in ids. For example, for the train file of icews14:
```
python ./data_utils/id_words.py --file_to_convert ./data/icews14/train.txt --path_output ./data/processed_new/icews14/train.txt --dataset icews14 --period 24
```
Rules learning parameters:
- **-f** **file_to_convert**, input path to a certain file.
- **-o** **--path_output**, output path to a certain file
- **-d** **--dataset**, icews14, icews18, GDELT or YAGO. 
- **-p** **--period**, default 1; to set 24: period for icews14/18 where timestamps increase every 24. 
Datasets containing all facts of train, valid and test are also provided (all_facts.txt)

By default you will create any new datasets of your own in ./data/processed_new/ .

### Rules learning
The rules learning part is originally from [Tlogic rules learning codes](https://github.com/liu-yushan/TLogic). 
It runs on lexical datasets (although it just convert them into ids). By default it only reaches datasets in ./data/ instead of ./data/processed_new/ . 
You can produce other rule banks besides the provided ones by running e.g. for icews14:
```
cd data_utils/rules_learning
python3 learn.py -d icews14 -l 1 2 3 -n 200 -p 15 -s 12
```
Rules learning parameters:
- **-d** **--dataset**, dataset name.
- **-l** **--rule_lengths**, default length of chains. (only length=1 is used)
- **-n** **--num_walks**
- **--transition_distr**, default: exp.
- **-p** **--num_processes**, for accelerating. 
- **-s** **--seed**

You will get a rule bank file similar to "060723022344_r[1,2,3]_n200_exp_s12_rules.json" under the ./output/ folder. 

### History retrieving

Find the file name of rule bank json (in ./output) and run from the folder TKGForcaster:
```
cd TKGForcaster
python3 ./data_utils/retrieve.py --name_of_rules_file name_rules.json --dataset icews14
```
An example for icews18 would be like:
```
python ./data_utils/retrieve.py --name_of_rules_file 060723022344_r[1]_n200_exp_s12_rules.json --dataset icews18
```
By default you will create these following files:
- data/processed_new/{dataset}/[train, valid, test]/history_facts/history_facts_{dataset}.txt [A]
- data/processed_new/{dataset}/[train, valid, test]/history_facts/history_facts_{dataset}_idx_fine_tune_all.txt
- data/processed_new/{dataset}/[train, valid, test]/test_answers/test_answers_{dataset}.txt [B]

For training, you need to convert history_facts files into lora json file:
```
python3 ./data_utils/create_json_train.py --dir_of_trainset 'the_full_trainset_to_convert (see [A])' --dir_of_answers 'the_test_answers (see [B])' --dir_of_entities2id 'the_json_of_entities2id (see [C])' --path_save 'better_the_same_as_the_trainset_before_converting'
```
An example for icews18 would be like:
```
python ./data_utils/create_json_train.py --dir_of_trainset './data/processed_new/icews18/train/history_facts/history_facts_icews18.txt' --dir_of_answers './data/processed_new/icews18/train/test_answers/test_answers_icews18.txt' --dir_of_entities2id './data/icews18/entity2id.json' --path_save './data/processed_new/icews18/train'
```

### Training
Basic training: 

```
python3 main.py --OUTPUT_DIR "your_output_directory" --DATA_PATH "path_of_dataset_file"
```
Example for training: 
```
python3 main.py --OUTPUT_DIR "./model/output_model_icews14_1024" --DATA_PATH "./data/processed/train/icews14/icews14_1024.json"
```
Training parameters (in config.py):

- **MICRO_BATCH_SIZE**, Per device train batch size.
- **BATCH_SIZE**, batch size.
- **EPOCHS**, Training epochs.
- **WARMUP_STEPS**, Warmup steps.
- **LEARNING_RATE**, Training learning rate.
- **CONTEXT_LEN**, Truncation length of context (in json).
- **TARGET_LEN**, Truncation length of target (in json).
- **TEXT_LEN**, Truncation length of text (in txt).
- **LORA_R**, Lora low rank.
- **LORA_ALPHA**, Lora Alpha.
- **LORA_DROPOUT**, Lora dropout.
- **MODEL_NAME**, Model name.
- **LOGGING_STEPS**, Logging steps in training.
- **LOAD_BEST_MODEL_AT_END**, set 1 to save the best checkpoint.
- **OUTPUT_DIR**, Output dir.
- **DATA_PATH**, Input dir of trainset.
- **DATA_TYPE**, Input trainsetfile type.
    
If you want to use logging platform like WandB, you may need these:

- **REPORT_TO**, logging to e.g. wandb.
- **PROJ_NAME**, Project name for e.g. wandb.
- **RUN_NAME**, Run name for e.g. wandb.
- **SAVE_STEPS**, Save the model according to steps.
- **SAVE_TOTAL_LIMIT**, The number of the checkpoint you will save (Excluding the final one).
- **W_RESUME**, set 1 to enable WANDB_RESUME.
- **W_ID**, set 1 to enable WANDB_RESUME'

### Test
Basic test: 

```
python3 inference.py --LORA_CHECKPOINT_DIR "path of model checkpoint" --output_file "your output directory" --input_file "path of history_facts file" --test_ans_file "path of test_answers file"
```
Example for testing: 
```
python3 main.py --LORA_CHECKPOINT_DIR "./model/icews14" --output_file "./results/prediction_icews14.txt"  --input_file "./data/processed/eval/history_facts/history_facts_icews14.txt"  --test_ans_file "./data/processed/eval/test_answers/test_ans_icews14.csv"
```
Testing parameters (in eval_utils.py):

- **LORA_CHECKPOINT_DIR**, path of model checkpoint.
- **output_file**, your output directory.
- **input_file**, path of history_facts file.
- **test_ans_file**, path of test_answers file.

If you want to begin from a certain i-th question (like resuming):
- **begin**, The number of the checkpoint you will save (Excluding the final one).
- **last_metric**, path for the saved metric file, it will read the results from it.

## File structure
### Repository

The repository contains codes for both tlogic retrieving and finetuning llama2 and inference. Learn rule banks are also provided here:
```
Root
|--data_utils/
	|--rules_learning/ (codes from [Tlogic](https://github.com/liu-yushan/TLogic))
	|--basic.py (utils for data reading/writing etc)
	|--create_json_train.py (convert dataset into lora json format)
	|--id_words.py (convert between id and lexical entities, relations and timestamps)
	|--retrieve.py (data reading/writing and so on for retrieving)
	|--TLR.py (retrieve history according to rules)
|--llama2_ori_repo/ (In-context Learning codes for llama2; imported in evaler.py)
|--minimal20b/ (In-context Learning codes for gpt-neox; imported in evaler.py)
|--output/ (contains rules banks from Tlogic rules learning)
|--results/ (stores inference results; empty)
	|--config.py
	|--eval_utils.py
	|--evaler.py
	|--inference.py (inference)
	|--main.py (training)
	|--neox.py (gpt-neox inference)
	|--utils.py
```
### Datasets

The structure should be similar like this:
```
Datasets
|--processed/
	|--train/ (trainsets for Gentkg; JSON files)
		|--icews14/
			|--icews14.json (full set)
			|--icews14_16.json (sampled set)
			...
			|--icews14_1024.json (sampled set)
		|--icews18/
		...
	|--eval/
		|--history_facts/
			|--history_facts_icews14.txt
			|--history_facts_icews18.txt
			|--history_facts_GDELT.txt
			|--history_facts_YAGO.txt
		|--test_answers/
			|--test_ans_icews14.csv
			|--test_ans_icews18.csv
			|--test_ans_GDELT.txt
			|--test_ans_YAGO.txt
|--original/ (original datasets mainly for rule based models)
	|--icews14/
		|--all_facts.txt
		|--train.txt
		|--valid.txt
		|--test.txt
		|--stat.txt
		|--entity2id.json (JSON as dictionary format; for GenTKG) [C]
		|--relation2id.json (JSON as dictionary format; for GenTKG)
		|--ts2id.json (JSON as dictionary format; for GenTKG)
	|--icews18/
		|--all_facts.txt
		|--train.txt
		|--valid.txt
		|--test.txt
		|--stat.txt
		|--entity2id.json
		|--relation2id.json
		|--ts2id.json
	...
```


## Data format
### Training json

```
{"context":question1, "target":answer1}{"context":question2, "target":answer2}...
```


### Files for testing 

The file format is as follows：

history_facts: 
```
history1.1
history1.2
history1.3
...
query1

history2.1
history2.2
history2.3
...
query2

...
...
```

test_ans: 
```
query_answer1
query_answer2
query_answer3
...
```



## Reference
[liu-yushan/TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs (github.com)](https://github.com/liu-yushan/TLogic)

[Fine_Tuning_LLama | Kaggle](https://www.kaggle.com/code/gunman02/fine-tuning-llama?scriptVersionId=128204744)

[FreddyBanana/ChatGLM2-LoRA-Trainer: Simple 4-bit/8-bit LoRA fine-tuning for ChatGLM2 with peft and transformers.Trainer. (github.com)](https://github.com/FreddyBanana/ChatGLM2-LoRA-Trainer)

[mymusise/ChatGLM-Tuning: An affordable chatgpt implementation solution, based on ChatGLM-6B + LoRA (github.com)](https://github.com/mymusise/ChatGLM-Tuning/tree/master)

## Citation

Please cite our work as follow if you find our work helpful.
@article{liao2023gentkg,
  title={GenTKG: Generative Forecasting on Temporal Knowledge Graph},
  author={Liao, Ruotong and Jia, Xu and Ma, Yunpu and Yangzhe Li and Tresp, Volker},
  journal={arXiv preprint arXiv:2310.07793},
  year={2023}
}
