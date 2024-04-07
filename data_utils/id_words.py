import argparse
from basic import read_json, flip_dict, str_dict, read_txt_as_list, write_txt, create_folder_for_file
def convert(str_in, dict_in):
    return dict_in[str_in]

def id_words(li, dict_ent, dict_r, dict_t, end=str(0), period=1):
    li_new = []
    for line in li:
        columns = line.strip().split('\t')
        columns[0] = str(convert(columns[0],dict_ent))
        columns[1] = str(convert(columns[1],dict_r))
        columns[2] = str(convert(columns[2],dict_ent))
        columns[3] = str(convert(str(int(columns[3])*period),dict_t))
        line = "\t".join([columns[0], columns[1], columns[2], columns[3], end])
        li_new.append(line)
    return li_new

def convert_dataset(li_to_convert, path_workspace, end=str(0), period=1):
    relations = read_json(path_workspace+'relation2id.json')
    entities = read_json(path_workspace+'entity2id.json')
    times_id = read_json(path_workspace+'ts2id.json')
    test_ans = id_words(li_to_convert, 
                        str_dict(flip_dict(entities)), 
                        str_dict(flip_dict(relations)),
                        str_dict(flip_dict(times_id)), end, period) #convert list in ids to list in words
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
    parsed = parser()
    file_to_convert = parsed["file_to_convert"]
    path_output = parsed["path_output"]
    create_folder_for_file(path_output)
    type_dataset = parsed["dataset"]
    period = parsed["period"]
    
    path_workspace = "./data/"+type_dataset+"/" 
    li_to_convert = read_txt_as_list(file_to_convert)
    test_ans = convert_dataset(li_to_convert, path_workspace, end='', period=period)
    write_txt(path_output, test_ans, head='')