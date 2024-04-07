import json
import csv
import os
import random
from pathlib import Path
import sys

def flip_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

def str_dict(original_dict):
    return {str(k): str(v) for k, v in original_dict.items()}

#2023/12/29
#author: Fowler (from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print)
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value
    return func_wrapper

def get_ins(): #ins for datasets; in json every " should be \"
    ins = '''You must be able to correctly predict the next {object_label} from a given text consisting of multiple quadruplets in the form of \"{time}:[{subject}, {relation}, {object_label}.{object}]\" and the query in the form of \"{time}:[{subject}, {relation},\" in the end.\nYou must generate {object_label}.{object}\n\n<</SYS>>'''
    head = "<s>[INST] <<SYS>>"
    return head + "" + ins #.replace('"', '\"')

def get_file_extension(file_path):
    _, extension = os.path.splitext(file_path)
    return extension    

def read_csv(csv_dir, col=None):
    with open(csv_dir, 'r', newline='', encoding='utf-8') as q:
        csv_data = csv.reader(q)
        if col is None:
            return [row for row in csv_data]
            # 得到所有 test questions, 以 quadruple 的形式. 
        else:
            return [row[col] for row in csv_data]

__author__ = "Yangzhe Li" #for the rest
__email__ = "yangzhe.li@tum.de"

def read_csv_as_dict(path_csv_file):
    csv_data = []
    with open(path_csv_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            csv_data.append({
                "space": row["space"],
                "underscore": row["underscore"]
            })
    return csv_data

def read_json(json_dir):
    with open(json_dir, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data

def read_json_as_list(json_dir):
    return list(read_json(json_dir).keys())

def just_read_txt(path_txt):
    with open(path_txt) as file:
        content = file.read()
    return content

def read_txt_as_list(path_txt):
    with open(path_txt, 'r', encoding='utf-8-sig') as file:
        data = file.readlines()
    return data

def read_txt_as_index_dict(path_txt, divider='\t'):
    li_corres = []
    li = read_txt_as_list(path_txt)
    for line in li:
        line_splited = line.strip().split(divider)
        li_corres.append({
                "space": line_splited[0],
                "underscore": line_splited[1]
            })
    return li_corres

def write_txt(txt_dir, out_list, head='\t'):
    with open(txt_dir, 'w', encoding='utf-8') as txtfile:
        for sublist in out_list:
            txtfile.write(head.join(map(str, sublist)) + '\n')

def write_dict(txt_dir, out_dict):
    with open(txt_dir, 'w', encoding='utf-8') as txtfile:
        for key, value in out_dict.items():
            txtfile.write(f"{key}\t{value}\n")

def write_csv(data, out_dict):
    with open(out_dict, 'w', encoding='utf-8') as out_dict:
        writer = csv.writer(out_dict)
        writer.writerow(['Column1','Column2','Column3','Column4'])
        for entry in data:
            out_dict.write(','.join(map(str, entry))+ '\n')

def just_write_json(data, out, indent=4):
    with open(out, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=indent)

def sample_dataset(dir_dataset, num_sample, is_json):
    path = Path(dir_dataset)
    with open(path, 'r', encoding='utf-8') as input_file:
        input_file = list(json.load(input_file)) if is_json else list(input_file)
        output_data = random.sample(input_file, num_sample) #attention: sampled data has no time order for queries
    sampled_dir_name = str(path.parent / path.stem) + "_" + str(num_sample) + path.suffix
    with open(sampled_dir_name, 'w', encoding='utf-8') as output_file:
        if is_json:
            json.dump(output_data, output_file, indent=4)
        else:
            output_file.writelines([f"{item}" for item in output_data])
    print("sampled", sampled_dir_name)
    return sampled_dir_name

def create_folder_for_file(file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)