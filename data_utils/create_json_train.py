from basic import just_write_json, just_read_txt, get_ins, read_txt_as_list, read_json
import random
import argparse
import os

def convert_txt_to_json(inputs, entities, test_ans):
    ins = get_ins()
    test_ans = [x.strip().split('\t') for x in test_ans]
    
    data_list = []
    for i in range(len(test_ans)):
        name_obj = test_ans[i][2]
        data = {
            "context": ins + inputs[i]+ "[/INST]",
            "target": str(entities[name_obj])+'.'+name_obj # top1: i+1: csv
        }
        data_list.append(data)
    return data_list

def sample_data_training(dir_dataset, dir_of_answers, dir_of_entities2id, path_save, name_train, nums_sample = [16, 54, 256, 512, 1024]):
    content = just_read_txt(dir_dataset)
    inputs = content.split('\n\n')
    test_ans = read_txt_as_list(dir_of_answers)
    entities = read_json(dir_of_entities2id)
    data_list = convert_txt_to_json(inputs, entities, test_ans)
    output_file_full = path_save+"/"+name_train+'.json'
    just_write_json(data_list, output_file_full, indent=4)
    print("saved as ", output_file_full)
    for num in nums_sample: 
        sampled_data = random.sample(data_list, num) # 16, 64, 256, 512, 1024
        output_file = path_save+"/"+name_train+'_' + str(num) + '.json'
        just_write_json(sampled_data, output_file, indent=4)
        print("saved as ", output_file)

def parse_int_list(string):
    try:
        return [int(x) for x in string.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list of integers: {}".format(string))
       
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_of_trainset", default="", type=str)
    parser.add_argument("--dir_of_answers", default="", type=str)
    parser.add_argument("--dir_of_entities2id", default="", type=str)
    parser.add_argument("--path_save", default="", type=str)
    parser.add_argument('--nums_sample', default="16", type=parse_int_list, 
                        help='The numbers of samples in a list of integers separated by commas')
    parser.add_argument("--name_train", default="icews14", type=str)
    parsed = vars(parser.parse_args())
    return parsed

if __name__ == "__main__":
    parsed = parser()
    dir_of_trainset = parsed["dir_of_trainset"]
    dir_of_answers = parsed["dir_of_answers"]
    dir_of_entities2id = parsed["dir_of_entities2id"]
    path_save = parsed["path_save"]
    nums_sample = parsed["nums_sample"]
    name_train = parsed["name_train"]

    if not os.path.exists(path_save):
            os.makedirs(path_save)
    sample_data_training(dir_of_trainset, dir_of_answers, dir_of_entities2id, path_save, name_train, nums_sample)