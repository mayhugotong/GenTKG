from transformers import LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

import argparse
import csv
import re
from data_utils.basic import get_file_extension

def read_results(path_results, divider=', \n'): #, \n for 18 yago llama2 or \n
    with open(path_results, 'r', encoding='utf-8') as file:
        content = file.read() #
    tests = content.split(', \n\n') #, \n\n for 18 yago llama2 or \n\n
    return [re.split(divider, test) for test in tests] 

def read_num_and_li_results(test):
    pattern_end = r'(.+?)( \n|"]})'   #'(.+?)( \n|"]})' for 18 yago llama2 or (.+?)(, |"]})
    num_Test = re.search(r'\d+', test[0]).group() if re.search(r'\d+', test[0]) else ""
    li_results = [re.match(pattern_end, answer).group(1) if re.match(pattern_end, answer) else answer \
                for answer in test[1:]] 
    return num_Test, li_results
    
def read_test_and_divide(path): #also in evaler.py
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    tests = content.split('\n\n')
    return tests

def read_test_an(pth_ans, col=2):
    file_type = get_file_extension(pth_ans)
    test_ans = []
    if file_type == ".csv":
        with open(pth_ans, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            test_ans = [row1[col] for row1 in reader] #take obj from [sub, rel, obj, time] as ans
            test_ans = test_ans[1:] # 
    else:
        with open(pth_ans, "r", encoding='utf-8') as f:
            lines = f.readlines()
        for i in range(len(lines)): 
            test_ans.append(lines[i].split('\t')[col])
    return test_ans

def read_last_metric(last_metric):
    if last_metric != '': 
        with open(last_metric, 'r') as file:

            lines = file.readlines()

        last_c_k = lines[-4:-1] # [-5:-2]

        last_c_k = [int( line.strip() ) for line in last_c_k]

        c1 = int(last_c_k[0])
        c3 = int(last_c_k[1])
        c10 = int(last_c_k[2])
        
    else: # 
        c1 = 0 # 
        c3 = 0
        c10 = 0
    # print('initial c1: ', c1)
    # print('initial c3: ', c3)
    # print('initial c10: ', c10 )

    return {"c1": c1, "c3": c3, "c10": c10} #c

def decide_model(args):
    if args.BIT_8: #
        model = LlamaForCausalLM.from_pretrained(
            args.MODEL_NAME,
            # load_in_8bit_fp32_cpu_offload=True,
            load_in_8bit=True,
            # torch_dtype=torch.load_in_8bit,
            # device_map="auto",
            trust_remote_code=True,
        )
    elif args.BIT_4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = LlamaForCausalLM.from_pretrained(
            args.MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
        )
    return PeftModel.from_pretrained(model, args.LORA_CHECKPOINT_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description="Config of Llama2-lora")

    # inference : 
    parser.add_argument('--MODEL_NAME', type=str, default="TheBloke/Llama-2-7B-fp16", help='Model name')
    parser.add_argument('--LORA_CHECKPOINT_DIR', type=str, default="", help='Your Lora checkpoint')
    parser.add_argument('--CONTEXT_LEN', type=int, default=4096, help='Truncation length of context (in json)')
    parser.add_argument('--BIT_8', default=True, action="store_true", help='Use 8-bit')
    parser.add_argument('--BIT_4', default=False, action="store_true", help='Use 4-bit')
    parser.add_argument('--TEMPERATURE', type=int, default=0, help='Temperature when inference')
    parser.add_argument('--PROMPT', type=str, default="Input your prompt", help='Your prompt when inference')
    parser.add_argument('--input_file', type=str, default="", help='Your history_facts file')
    parser.add_argument('--output_file', type=str, default="", help='Output text prediction')
    parser.add_argument('--test_ans_file', type=str, default="", help='Your ground truth file')
    parser.add_argument('--fulltest', type=str, default="", help='fulltest with dense quadruples. For whole set filtering')
    parser.add_argument('--time2id', type=str, default="", help='time2id json file. For whole set filtering')
    parser.add_argument('--begin', type=int, default=0, help='Where to continue. default to -1')
    parser.add_argument('--max_gen_len', type=int, default=27, help='18 for Gdelt 27 for Yago; 27 as default')
    parser.add_argument('--max_seq_len', type=int, default=30, help='4096 for llama2 icl; 30 as default')
    parser.add_argument('--last_metric', type=str, default="", help='Last metric result *file* when interrupted. ')
    parser.add_argument('--FILTER', type=int, default=1, help='Set 1 to filter multiple objects. ')
    parser.add_argument('--path_results', type=str, default="", help='Path of the result file to be filtered. ')
    parser.add_argument('--ft', type=int, default=1, help='Set 0: no finetuned model. ')
    parser.add_argument('--local-rank', type=int, default=0, help='for torch.distributed.launch. ')
    parser.add_argument('--instruct_yes', type=int, default=0, help='Set 0 to give no instruction in the pompts. ')
    
    return parser.parse_args()