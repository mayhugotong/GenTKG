from TLR import Retriever
from basic import read_txt_as_list, read_json, write_txt
from id_words import convert_dataset
import os, glob
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="icews14", type=str)
    parser.add_argument("--retrieve_type", "-t", default="TLogic-3", type=str)
    parser.add_argument("--name_of_rules_file", "-r", default="", type=str)
    parsed = vars(parser.parse_args())
    return parsed
    
if __name__ == "__main__":
    parsed = parser()
    retrieve_type = parsed["retrieve_type"]
    type_dataset = parsed["dataset"]
    name_rules = parsed["name_of_rules_file"]
    
    path_workspace = "./data/"+type_dataset+"/" #Icew s14 /icews14
    path_out_tl = "./output/"+type_dataset+"/"
    print(path_out_tl)
    
    path_save = "./data/processed_new/"+type_dataset+"/"
    if not os.path.exists(path_save):
            os.makedirs(path_save)
        
    period = 1
    if type_dataset == "icews18":
        num_relations = 256 #for ICEWS18 #set before np.array
        period = 24
    elif type_dataset == "icews14":
        num_relations = 230
        period = 24
    elif type_dataset == "GDELT":
        num_relations = 238 #GDELT and
    else:
        num_relations = 24 # YAGO
        
    test_ans = []
    
    #open files:
        
    li_files = ['train','valid','test']#  ['train','valid','test'] or  ['test'] when only test set is needed
    
    for files in li_files:
        print("exiting rules:", glob.glob(path_out_tl+'*rules.json'))
        dir_rules = glob.glob(path_out_tl+'*rules.json')[0] if name_rules=="" else path_out_tl+name_rules
        print("files", files)
        test_ans = read_txt_as_list(path_workspace+files+'.txt')
        
        relations = read_json(path_workspace+'relation2id.json')
        entities = read_json(path_workspace+'entity2id.json')
        times_id = read_json(path_workspace+'ts2id.json')
        
        test_ans = convert_dataset(test_ans, path_workspace, period=period)
        
        chains = read_json(path_out_tl+name_rules)
        rel_keys = list(relations.keys())
        ent_idx = list(entities.keys()) # [0, 1, ...]
        times_id_keys = list(times_id.keys())
        all_facts = []
        with open(path_workspace+"all_facts.txt", "r", encoding='utf-8') as f:
            all_facts = f.readlines()
        
        rtr = Retriever(test_ans, all_facts, entities, relations, times_id, num_relations, chains, rel_keys, dataset=type_dataset)
        test_idx, test_text = rtr.get_output()
                
        path_file = path_save+files+"/history_facts/"+"history_facts_"+type_dataset #"history_facts_"+retrieve_type+type_dataset
        path_file_word = path_file+".txt"
        path_file_id = path_file+"_idx_fine_tune_all.txt"
        
        if not os.path.exists(path_save+files+"/history_facts/"):
            os.makedirs(path_save+files+"/history_facts/")
        write_txt(path_file_id, test_text)
        with open(path_file_word, 'w', encoding='utf-8') as f:
            for i in range(len(test_text)): 
                f.write(test_text[i][0] + '\n')
        print("saved as ", path_file_word, "and ", path_file_id)
        
        path_answer = path_save+files+"/test_answers/"+"test_answers_"+type_dataset+".txt"
        if not os.path.exists(path_save+files+"/test_answers/"):
            os.makedirs(path_save+files+"/test_answers/")
        write_txt(path_answer, test_ans, head='')
        print("saved as ", path_answer)
