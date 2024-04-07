from transformers import LlamaTokenizer
import torch
import re

import os
from evaler import Evaler
from eval_utils import parse_args, read_test_and_divide, read_test_an, read_last_metric, decide_model

if __name__ == "__main__":
    
    args = parse_args()

    eval_txt_path = args.output_file[:-4]+'_metric_results.txt'

    test_ans = read_test_an(args.test_ans_file)
    if args.ft == 1:    
        model = decide_model(args)    
        tokenizer = LlamaTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)
    else:
        args.max_batch_size = 9 
        model = None
        tokenizer = None
        print('Using unfinetuned model ')
    tests = read_test_and_divide(args.input_file)

    c = read_last_metric(args.last_metric) #dict with keys: c1 c3 c10

    pattern1 = re.compile(r'.*?[\d:@][._](.*?)[\]\[]?([< ].*?)?$') 
    pattern2 = re.compile(r'<s> .*?[\n]?([A-Z\u00C0-\u00DD\u0388-\u03AB\u0410-\u042F\u0600-\u06FF\u4e00-\u9fa5].*)\]')
    pattern3 = re.compile(r'<s> *(.*)\]') 
    is_with_id = True
    if is_with_id:
        patterns = [pattern1]
    else:
        patterns = [pattern1, pattern2, pattern3]
    topk= 10
    cnt = args.begin 
    early_stop_chars = [torch.tensor([29962], device='cuda:0'), #]
                        torch.tensor([29961], device='cuda:0'), #[
                        torch.tensor([4638], device='cuda:0'),  #)]
                        torch.tensor([29871], device='cuda:0')]  # 
    obligations = []
    evaler = Evaler(topk, tests, test_ans, eval_txt_path, args, model, tokenizer, patterns, early_stop_chars, obligations)
    
    path_results = args.path_results
    path_results = os.path.normpath(path_results)

    if path_results != '.':
        evaler.eval(c, cnt, path_results)
    else:
        evaler.eval(c, cnt, filter_yes=args.FILTER)