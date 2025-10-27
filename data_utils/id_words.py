import argparse
from basic import str_dict, read_txt_as_list, write_txt, create_folder_for_file
import json
def convert(str_in, dict_in):
    return dict_in[str_in]

def id_words(li, dict_ent, dict_r, dict_t, end=str(0), period=1):
    li_new = []
    for line in li:
        columns = line.strip().split('\t')
        print(columns)
        columns[0] = str(convert(columns[0],dict_ent))
        columns[1] = str(convert(columns[1],dict_r))
        columns[2] = str(convert(columns[2],dict_ent))
        columns[3] = str(convert(str(int(columns[3])*period),dict_t))
        line = "\t".join([columns[0], columns[1], columns[2], columns[3], end]) \
            if end != '' else "\t".join([columns[0], columns[1], columns[2], columns[3]])
        li_new.append(line)
    return li_new

def read_mapping_file(path):
    if path.endswith(".json") or path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif path.endswith(".txt"):
        mapping = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    name, idx = parts
                    mapping[name] = int(idx)
        return mapping
    else:
        raise ValueError(f"Unsupported file format: {path}")

def convert_dataset(li_to_convert, path_workspace, end=str(0), period=1, type_file='.txt'):
    relations = read_mapping_file(path_workspace + 'relation2id' + type_file)
    entities = read_mapping_file(path_workspace + 'entity2id' + type_file)
    times_id = read_mapping_file(path_workspace + 'ts2id' + type_file)
    
    relations = {v: k for k, v in relations.items()}
    entities = {v: k for k, v in entities.items()}
    times_id = {v: k for k, v in times_id.items()}

    test_ans = id_words(
        li_to_convert,
        str_dict(entities),
        str_dict(relations),
        str_dict(times_id),
        end,
        period
    )

    return test_ans

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_to_convert", "-f", default="", type=str, 
                        help="input path; you may set as ./data/icews14/[train|valid|test].txt or something")
    parser.add_argument("--path_output", "-o", default="", type=str, 
                        help="output path; you may set as ./data/processed_new/icews14/train/[train|valid|test].txt or something")
    parser.add_argument("--dataset", "-d", default="", type=str)
    parser.add_argument("--period", "-p", default=1, type=int, 
                        help="default 1; to set 24: period for icews14/18 where timestamps increase every 24")
    parsed = vars(parser.parse_args())
    return parsed
    
if __name__ == "__main__":
    import os
    parsed = parser()
    file_to_convert = parsed["file_to_convert"]
    path_output = parsed["path_output"]
    create_folder_for_file(path_output)
    type_dataset = parsed["dataset"]
    period = parsed["period"]
    
    # path_workspace = "./data/"+type_dataset+"/" 
    path_workspace = os.path.dirname(file_to_convert).replace("\\", "/") + "/"
    li_to_convert = read_txt_as_list(file_to_convert)
    test_ans = convert_dataset(li_to_convert, path_workspace, end='', period=period)
    write_txt(path_output, test_ans, head='')
    print("finished writing to "+path_output)