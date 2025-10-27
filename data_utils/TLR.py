import numpy as np
from basic import flip_dict
import time as ti
from tqdm import tqdm
#use lexicon datasets
class Retriever:
    def __init__(self, 
                 test, all_facts, 
                 entities, relations, times_id, 
                 num_relations, chains, rel_keys, dataset, retrieve_type='TLogic',):
        self.retrieve_type = retrieve_type
        self.dataset = dataset
        self.test = test
        self.all_facts = all_facts
        
        self.entities = entities
        self.relations = relations
        self.times_id = times_id
        self.num_relations = num_relations
        self.chains = chains
        
        self.entities_flip = flip_dict(self.entities)
        self.relations_flip = flip_dict(self.relations)
        col_sub = []
        col_rel = []
        col_obj = []
        col_time = []
        for row in all_facts:
            row = row.strip().split('\t')
            col_sub.append(row[0]) #take sub
            col_rel.append(row[1]) #Get the relation column of all facts
            col_obj.append(row[2]) #Take obj
            col_time.append(row[3]) #Time in Str form
        self.col_obj = np.array(col_obj) #To get cand and then search for facts use
        self.col_sub = np.array(col_sub) #To get cand and then search for facts use
        self.col_time = np.array(col_time) #time, str form
        self.col_rel = np.array(col_rel) #relation, str form
        all_facts_array = np.array(all_facts)
        
        self.rel_keys = np.array(rel_keys)
        
    def prepare_bs(self, i):
        sub, rel, _, time, _ = self.test[i].strip().split("\t")
        idx_t = np.where(self.col_time < time)[0] #cannot be equal to
        s_t = set(idx_t)
        idx0 = np.where(self.col_sub == sub)[0]
        s0 = set(idx0)
        idx = list(s0 & s_t)
        idx.sort(reverse=True)
        time = self.times_id[time]
        return time, sub, rel, idx
    
    def build_bs(self):
        #Pure Entity mode
        test_text = []
        test_idx = []

        for i in tqdm(range(0, len(self.test))): #csv has a header, txt has no header. len(self.test)
            num_facts = 50 #20 or 100
            time, sub, rel, idx = self.prepare_bs(i)

            facts = []
            idx = idx[0:num_facts]
            for k in idx:
                facts.append(self.all_facts[k]) #Get the facts where sub and rel are the same

            if len(facts)<num_facts: 
                num_facts = len(facts)

            histories = self.collect_hist(i, facts, num_facts)
            history_query = self.build_history_query(time, sub, rel, histories=histories)

            test_idx.append(idx)
            test_text.append(history_query)
        return test_idx, test_text
    
    def tlogic_prepro(self, i):
        # test_sub, test_rel, _, test_time, _ = self.test[i].strip().split("\t") for dataset with extra column
        test_sub, test_rel, _, test_time = self.test[i].strip().split("\t")
        #First of all, there must be a time premise of retrieve s_t
        #Here we need to find out the idx of test in all_facts so that it can be removed
        idx_test = len(self.all_facts)- (len(self.test)-1) + i -1
        # #The major premise is that retrieval must be performed from those ranges earlier than test_time
        idx_t = np.where(self.col_time < test_time)[0] 
        s_t = set(idx_t)
        if idx_test in s_t:
                s_t.remove(idx_test) #Remove the test item itself
                #Second, the search for the beginning of the chain needs to be restricted to rel==test_sub
        idx_test_sub = np.where(self.col_sub == test_sub)[0]
        s_test_sub = set(idx_test_sub)
        s_0 = s_t & s_test_sub #Get: major premise
        head_rel = self.relations[test_rel] #Get the idx corresponding to test_relation: 0,1,2,...
        time = self.times_id[test_time] #To move forward the time according to the id corresponding relationship of time. Get int from str
        return s_0, s_t, head_rel, time, test_sub, test_rel
    
    def build_tl(self):
        test_text = []
        test_idx = []

        for i in tqdm(range(0,  len(self.test))): #Starting from 1 because there is a header. 1, len(test)
            num_facts = 50 #Set it again for each question, as the following may change.
            s_0, s_t, head_rel, time, test_sub, test_rel = self.tlogic_prepro(i)
            facts = []
            idx = []

            if not str(head_rel) in self.chains: #If the test relation has no chain, nothing will be done.
                #l = ['Just repeat "No Chains."\n']
                history_query = [str(int(time))+': ['+ test_sub +', '+ test_rel+',\n']
                test_idx.append([]) #At this time idx is a blank line
                test_text.append(history_query)
                #print(i, 'no chain in this line')
                continue
            s_0 = np.array(list(s_0))  #Convert collection to NumPy array
            #After the above preparations, start searching for facts by chain.
            idx_chain = []
            for k in range( 0,len(self.chains[str(head_rel)]) ): #There are len(chains[str(head_rel)]) chains
                body_rel_len = len( self.chains[str(head_rel)][k]['body_rels'] ) #The length of this chain
                if body_rel_len==1:
                    idx_chain.append(k)
            for k in idx_chain: #TLogic (or TLogic-3), as long as the shortest len=1 chain
                idx_case = []
                body_rel_len = len( self.chains[str(head_rel)][k]['body_rels'] ) #The length of this chain
                rel = self.chains[str(head_rel)][k]['body_rels'][-1] %self.num_relations 
                idx_rel = np.where(self.col_rel == self.rel_keys[rel])[0]
                idx_rel = np.intersect1d(idx_rel, s_0)  #Using NumPy functions for intersection operations
                idx_case = idx_rel
                idx_case.tolist()
                if len(idx_case) != 0: #If it is not empty, retrieve it.
                    idx_case = list( set(idx_case) )
                    idx = list( set( idx + idx_case) )
                else: 
                    continue #If no such chain exists, jump to the next chain
                if len(idx)>=num_facts: 
                    break #Break out of the loop on chains and go to the next test
            #time reordering
            idx.sort(reverse=True)
            #Idx with chain.sort(reverse=true)
            if len(idx)>num_facts:
                idx = idx[0:num_facts] 
                for a in idx:         
                    facts.append(self.all_facts[a])
            else: 
                for a in idx:         
                    facts.append(self.all_facts[a])
            test_idx.append(idx)
            if len(facts)==0: #If the test relation has no chain, nothing will be done.
                history_query = self.build_history_query(time, test_sub, test_rel) #Self.times id[time]
                test_text.append(history_query)
                #test_idx.append([]) The above line has been appended
                continue
            if num_facts >= len(facts):
                num_facts = len(facts)
                
            histories = self.collect_hist(i, facts, num_facts)
            history_query = self.build_history_query(time, test_sub, test_rel, histories=histories)
            test_text.append(history_query)
            ti.sleep(0.001)
        return test_idx, test_text

    def collect_hist(self, i, facts, num_facts):
        period = 1
        if self.dataset == "icews14" or self.dataset == "icews18":
            period = 24
        histories = []
        facts = facts[0:num_facts] # 
        facts.reverse() #Replace the order so that the last output is the one closest in time.
        for b in range(num_facts): 
            fact = facts[b].strip().split('\t')
            time_in_id = self.times_id[fact[3]]
            sub_in_word = fact[0]
            rel_in_word = fact[1]
            
            obj_in_word = fact[2]
            id_obj = self.entities[obj_in_word]
            histories= histories+ [str(int(time_in_id/period))
                    +': [' + sub_in_word +', ' + rel_in_word +', ' 
                    + str(id_obj)+'.'+ obj_in_word +'] \n']
        return histories

    def build_history_query(self, time, test_sub, test_rel, histories=''):
        period = 1
        if self.dataset == "icews14" or self.dataset == "icews18":
            period = 24
        # time_in_id = self.times_id[time]
        return [''.join(histories)  + str(int(time/period))+': ['+ test_sub +', '+ test_rel+',\n'#times id[time]
                ]
    
    def call_function(self, func_name):
        func = getattr(self, func_name)
        if func and callable(func):
            test_idx, test_text = func()
        else:
            print("Retrieve function not found")
        return test_idx, test_text
    
    def get_output(self):
        type_retr = "bs" if self.retrieve_type == 'bs' else "tl"
        test_idx, test_text = self.call_function("build_"+type_retr)
        
        return test_idx, test_text
