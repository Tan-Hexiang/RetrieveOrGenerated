from .evaluation import ems, has_answer, SimpleTokenizer,compute_ppl_from_logprobs, document_question_bertscore_similarity,  document_question_jaccard_similarity
from .jsonl import load_all_jsonl, dump_all_jsonl
from .data import find_with_question
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt




class context_conflicting_dataset:
    def __init__(self, com_path='', gen_path='', ir_path='', llm_path='', prefix={}, load_from_dict=None):
        self.attr_json = ['data_com','data_gen','data_ir','data_aig','data_air','data_full','data_full_wo_unk','data_full_w_unk','data_full_wo_unk_incontext','statistic','preference','em','prefix']
        self.prefix = prefix
        self.path = (com_path, gen_path, ir_path, llm_path)
        self.data_com, self.data_gen, self.data_ir = [], [], []
        # air: answer in retrieved context  aig: answer in generated context
        self.data_air, self.data_aig = [], []
        self.data_full = []
        self.data_full_wo_unk = [] # gen_ans and ir_ans are not unknown
        self.data_full_w_unk = [] # one of them is unknown
        self.data_full_wo_unk_incontext = [] # answer string in the input context
        '''
        question,
        answer [],
        generated_passage ,
        retrieved_passage ,
        gen_ans ,
        ir_ans ,
        com_ans

        gen_em
        ir_em
        com_em
        '''
        # metric
        self.statistic = {
            "unk_ratio_gen": -1,
            "unk_ratio_ir": -1,
            "num_not_in_context_gen":-1,
            "num_not_in_context_ir":-1,
            "tt":-1,
            "ff":-1,
        }
        self.preference = {
            "data_aig": {},
            "data_air": {},
        }
        self.em = {
            "data_full":{},
            "data_aig":{},
            "data_air":{},
        }
        # call main function
        if load_from_dict==None:
            self.pipeline()
        else:
            # print("Load from {}".format(load_from_dict))
            self.load_all_from_dict(load_from_dict)
        
    def info_construction(self):
        message = "\n ---------------prefix:{}-----------\constructiing:".format(self.prefix)
        message += "data_full: {}\n".format(len(self.data_full))
        message += "data_full_wo_unk: {}, data_full_w_unk: {} # unk_ratio_gen: {}, unk_ratio_ir: {}\n".format(len(self.data_full_wo_unk), len(self.data_full_w_unk), self.statistic['unk_ratio_gen'], self.statistic['unk_ratio_ir'])
        message += "data_full_wo_unk_incontext: {} # num_not_in_context_gen: {} num_not_in_context_ir: {}\n".format(len(self.data_full_wo_unk_incontext), self.statistic['num_not_in_context_gen'], self.statistic['num_not_in_context_ir'])
        message += "data_aig: {}, data_air: {}, # tt: {}, ff: {}\n".format(len(self.data_aig), len(self.data_air), self.statistic['tt'], self.statistic['ff'])
        print(message)
    
    def info_metric(self):
        print("\n---------------prefix:{}-----------\nMetric:\n {}\n{}\n".format(self.prefix,self.preference, self.em))

    def pipeline(self):
        # main function
        self._load_data()
        self._merge_data()
        self._filter_unk()
        self._check_answer_in_context()
        self._filter_parametric_knowledge()
        self._split_conflict_type()
        
        self.info_construction()
        # metric
        self.preference['data_aig'] = self._compute_preference(self.data_aig)
        self.preference['data_air'] = self._compute_preference(self.data_air)

        self.em['data_full'] = self._compute_em(self.data_full)
        self.em['data_aig'] = self._compute_em(self.data_aig)
        self.em['data_air'] = self._compute_em(self.data_air)
        # output metric
        self.info_metric()
    
    def analyze_similarity(self, device = 0):
        # compute similarity(bertscore or jaccard similarity)
        # input： self.data_aig， self.data_air
        # output：similarity key as follows
        '''
        "similarity":{
            "generated_passage": {
                "bertscore":[],
                "jaccard":[]
            }
            "retrieved_passage": {
                "bertscore":[],
                "jaccard":[]
            }
        }
        '''
        assert self.data_aig != []
        assert self.data_air != []
        # self.data_aig['similarity'], self.data_air['similarity'] = {}, {}
        # 计算jaccard
        for passage_name in ['generated_passage', 'retrieved_passage']:
            for k,_ in enumerate(self.data_aig):
                if 'similarity' not in self.data_aig[k]:
                    self.data_aig[k]['similarity'] = {"generated_passage": {"bertscore":[],"jaccard":[]},"retrieved_passage": {"bertscore":[],"jaccard":[]}}
                self.data_aig[k]['similarity'][passage_name]['jaccard'] = document_question_jaccard_similarity(self.data_aig[k][passage_name], self.data_aig[k]['question'])
            for k,_ in enumerate(self.data_air):
                if 'similarity' not in self.data_air[k]:
                    self.data_air[k]['similarity']= {"generated_passage": {"bertscore":[],"jaccard":[]},"retrieved_passage": {"bertscore":[],"jaccard":[]}}
                self.data_air[k]['similarity'][passage_name]['jaccard'] = document_question_jaccard_similarity(self.data_air[k][passage_name], self.data_air[k]['question'])
        print("Complete jaccard\nExample:{}".format(self.data_air[0]))
        # bertscore
        # for data_name in ['self.data_aig', 'self.data_air']:
        for passage_name in ['generated_passage', 'retrieved_passage']:
            passage_list, question_list = [],[]
            for k,example in enumerate(self.data_aig):
                passage_list.append(example[passage_name])
                question_list.append(example['question'])
            # compute bertscore
            all_bertscore = document_question_bertscore_similarity(passage_list, question_list, device)
            # print(len(all_bertscore), len(self.data_aig))
            assert len(all_bertscore) == len(self.data_aig)
            # save
            for k,_ in enumerate(self.data_aig):
                self.data_aig[k]['similarity'][passage_name]['bertscore'] = all_bertscore[k]
                # print(all_bertscore[k])
        
        for passage_name in ['generated_passage', 'retrieved_passage']:
            # convert to list
            passage_list, question_list = [],[]
            for k,example in enumerate(self.data_air):
                passage_list.append(example[passage_name])
                question_list.append(example['question'])
            # compute bertscore
            all_bertscore = document_question_bertscore_similarity(passage_list, question_list, device)
            # print(len(all_bertscore), len(self.data_aig))
            assert len(all_bertscore) == len(self.data_air)
            # save
            for k,_ in enumerate(self.data_air):
                # print(all_bertscore[k])
                self.data_air[k]['similarity'][passage_name]['bertscore'] = all_bertscore[k]
        print("Complete bertscore\nExample:{}".format(self.data_aig[0]))

    def save_all_to_dict(self, output_dict):
        if os.path.exists(output_dict) == False:
            os.makedirs(output_dict,exist_ok=True)
        for name in self.attr_json:
            data_to_save = getattr(self,name)
            if isinstance(data_to_save, dict):
                dump_all_jsonl(data = [data_to_save], output_path=output_dict+'/'+name+'.jsonl', append=False)
            else:
                dump_all_jsonl(data = data_to_save, output_path=output_dict+'/'+name+'.jsonl', append=False)
    
    def load_all_from_dict(self, saved_dict):
        for name in self.attr_json:
            data = load_all_jsonl(saved_dict+'/'+name+'.jsonl')
            raw_attr = getattr(self,name)
            if isinstance(raw_attr,dict):
                setattr(self, name, data[0])
            else:
                setattr(self, name, data)
            
    def get_generation_ratio(self, mode='sub'):
        if mode == 'sub':
            generation_ratio ={
                'data_aig':(self.preference['data_aig']['Gen-Ans']-self.preference['data_aig']['IR-Ans']) / (self.preference['data_aig']['Gen-Ans']+self.preference['data_aig']['IR-Ans']),
                
                'data_air':(self.preference['data_air']['Gen-Ans'] - self.preference['data_air']['IR-Ans'] )/ (self.preference['data_air']['Gen-Ans']+self.preference['data_air']['IR-Ans'])
            }
        else:
            print("Error mode, must be 'div' or 'sub'")
            exit()
        return generation_ratio
        
    
    def save_preference_to_csv(self, output_path, include_full_dataset=False):
        # Dataset	LLMs  Generator  Reader  Subset(Dataset-AIR, Dataset-AIG)	 Preference(Gen-Ans, IR-Ans, Others)	Preference Ratio
        suffix = ''
        Dataset = self.prefix['Dataset']
        LLMs = self.prefix['LLMs']
        Generator = self.prefix['Generator']
        Reader = self.prefix['Reader']
        result = []
        for pre in ['Gen-Ans', 'IR-Ans', 'Others']:
            result.append(
                {
                    'Dataset':Dataset,
                    'LLMs':LLMs,
                    'Generator': Generator,
                    'Reader':Reader,
                    'Subset':Dataset+'-'+'AIG',
                    'Preference':pre,
                    'Preference Ratio':self.preference['data_aig'+suffix][pre]
                }
            )
            result.append(
                {
                    'Dataset':Dataset,
                    'LLMs':LLMs,
                    'Generator': Generator,
                    'Reader':Reader,
                    'Subset':Dataset+'-'+'AIR',
                    'Preference':pre,
                    'Preference Ratio':self.preference['data_air'+suffix][pre]
                }
            )
            if include_full_dataset:
                full_preference = self._compute_preference(self.data_aig+self.data_air)
                result.append(
                {
                    'Dataset':Dataset,
                    'LLMs':LLMs,
                    'Generator': Generator,
                    'Reader':Reader,
                    'Subset':Dataset+'-'+'FULL',
                    'Preference':pre,
                    'Preference Ratio':full_preference[pre]
                }
            )
        main_df = pd.DataFrame(result)
        main_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
        return result

    def _compute_preference(self, result):
        gen_ans, ir_ans, llm_ans, Others = 0, 0, 0, 0
        total = len(result)
        if total == 0:
            return {'len': 0}
        for item in result:
            if ems(item['com_ans'], [item['ir_ans']]):
                ir_ans+=1
            elif ems(item['com_ans'], [item['gen_ans']]):
                gen_ans+=1
            elif ems(item['com_ans'], [item['llm_ans']]):
                llm_ans+=1
            else:
                Others+=1
        return {'Gen-Ans':gen_ans/total, 'IR-Ans':ir_ans/total, 'LLM-Ans':llm_ans/total, 'Others': Others/total, 'len': len(result)}

    def _compute_em(self,result):
        gen_em, ir_em, com_em = [], [], []
        total = len(result)
        if total == 0:
            return {'len': 0}
        for item in result:
            com_em.append(ems(item['com_ans'], item['answer']))
            gen_em.append(ems(item['gen_ans'], item['answer']))
            ir_em.append(ems(item['ir_ans'], item['answer']))
            # if ems(item['contriever_response'][0], item['answer']) == False:
                # print("\n", item['contriever_response'][0], item['answer'])
        return {'gen-em':sum(gen_em)/total, 'ir-em':sum(ir_em)/total, 'com-em': sum(com_em)/total, 'len': len(result)}

    def _load_data(self):
        try:
            self.data_com = load_all_jsonl(self.path[0])
        except:
            print("path error! paths = {}".format(self.path[0]))
        try:    
            self.data_gen = load_all_jsonl(self.path[1])
        except:
            print("path error! paths = {}".format(self.path[1]))
        try:
            self.data_ir = load_all_jsonl(self.path[2])
        except:
            print("path error! paths = {}".format(self.path[2]))
        try:
            self.data_llm = load_all_jsonl(self.path[3])
        except:
            print("path error! paths = {}".format(self.path[3]))
        
        print("len gen: {} ir: {} com: {}".format(len(self.data_gen), len(self.data_ir), len(self.data_com)))
        assert len(self.data_gen) == len(self.data_com)
        assert len(self.data_ir) == len(self.data_gen)

    def _merge_data(self):

        for example in self.data_com:
            question = example['question']
            # match question, because their order may be different
            item_gen = find_with_question(question, self.data_gen)
            item_ir = find_with_question(question, self.data_ir)
            try:
                merge_example = {
                    'question': example['question'],
                    'answer': example['answer'],
                    "generated_passage": example['generated_passage'][0]  ,
                    "retrieved_passage": example['retrieved_passage'][0] ,
                    "gen_ans": item_gen['response'][0] ,
                    "ir_ans": item_ir['response'][0] ,
                    "com_ans": example['response'][0],
                    "gen_em": ems(item_gen['response'][0], example['answer']),
                    "ir_em": ems(item_ir['response'][0], example['answer']),
                    "com_em": ems(example['response'][0], example['answer'])
                }
            except:
                print("Merge error, example:{}".format(example))
                print(len(self.data_com), len(self.data_gen), len(self.data_ir))
                print(self.prefix)
                exit()
            if ('logprobs' in example) and ('logprobs' in item_gen) and ('logprobs' in item_ir):
                merge_example.update(
                    {
                        "gen_ans_logprobs": item_gen['logprobs'] ,
                        "ir_ans_logprobs": item_ir['logprobs'] ,
                        "com_ans_logprobs": example['logprobs'],
                        "gen_ans_ppl": compute_ppl_from_logprobs(item_gen['logprobs']) ,
                        "ir_ans_ppl": compute_ppl_from_logprobs(item_ir['logprobs']) ,
                        "com_ans_ppl": compute_ppl_from_logprobs(example['logprobs']),
                        "gen_ans_tokens": item_gen['tokens'] ,
                        "ir_ans_tokens": item_ir['tokens'] ,
                        "com_ans_tokens": example['tokens'],
                    }
                )
            self.data_full.append(merge_example)

        print('Merge data and get data_full, example'.format(self.data_full[0]))        
        print('Len of data_full: {}'.format(len(self.data_full)))
    
    def _filter_unk(self):
        assert self.data_full !=[]
        def is_unk(candidate):
            if has_unknown_keyword(candidate) or ems(candidate, unknown_list):
                return True
            else:
                return False
        total = len(self.data_full)
        num_unk_gen, num_unk_ir = 0, 0 
        for example in self.data_full:
            if is_unk(example['gen_ans']) or is_unk(example['ir_ans']):
                self.data_full_w_unk.append(example)
            else:
                self.data_full_wo_unk.append(example)
            if is_unk(example['gen_ans']):
                num_unk_gen+=1
            if is_unk(example['ir_ans']):
                num_unk_ir+=1
        
        self.statistic['unk_ratio_gen'] = num_unk_gen/total
        self.statistic['unk_ratio_ir'] = num_unk_ir/total
        print("Filter out unk: data_full {}, data_full_wo_unk {}, data_full_w_unk {}".format(total, len(self.data_full_wo_unk), len(self.data_full_w_unk)))
        print(self.statistic)
    
    def _check_answer_in_context(self):
        assert self.data_full_wo_unk != []
        num_not_in_gen, num_not_in_ir = 0, 0
        tokenizer = SimpleTokenizer()
        for example in self.data_full_wo_unk:
            if has_answer(example['gen_ans'], example['generated_passage'], tokenizer) and has_answer(example['ir_ans'], example['retrieved_passage'], tokenizer):
                self.data_full_wo_unk_incontext.append(example)
            if not has_answer(example['gen_ans'], example['generated_passage'], tokenizer):
                num_not_in_gen+=1
            if not has_answer(example['ir_ans'], example['retrieved_passage'], tokenizer):
                num_not_in_ir+=1
        self.statistic['num_not_in_context_gen'] = num_not_in_gen
        self.statistic['num_not_in_context_ir'] = num_not_in_ir
        print("Check condidate ans in context: data_full_wo_unk {} --> data_full_wo_unk_incontext {}".format(len(self.data_full_wo_unk), len(self.data_full_wo_unk_incontext)))
        print("Statistics: {}".format(self.statistic))
    
    def _filter_parametric_knowledge(self):
        self.data_full_filtered  = []
        for item in self.data_full_wo_unk_incontext:
            llm_item = find_with_question(question=item['question'], data=self.data_llm)
            check_llm_gen = ems(llm_item['response'][0], [item['gen_ans']])
            check_llm_ir = ems(llm_item['response'][0], [item['ir_ans']])
            if check_llm_gen == False and check_llm_ir == False:
                item['llm_ans'] = llm_item['response'][0]
                self.data_full_filtered.append(item)
        print("Original data_full_wo_unk_incontext: {}".format(len(self.data_full_wo_unk_incontext)))
        print("data_full_filtered: {}".format(len(self.data_full_filtered)))
                
        
        
    def _split_conflict_type(self):
        assert self.data_full_filtered !=[]
        assert self.data_aig==[] and self.data_air==[]
        tt, ff = 0,0
        for example in self.data_full_filtered:
            if example['gen_em'] == True and example['ir_em'] ==False:
                self.data_aig.append(example)
            elif example['gen_em'] == False and example['ir_em'] == True:
                self.data_air.append(example)
            elif example['gen_em'] == True and example['ir_em'] == True:
                tt+=1
            elif example['gen_em'] == False and example['ir_em'] == False:
                ff+=1
        self.statistic['tt'] = tt
        self.statistic['ff'] = ff
        print("Split conflict type: data_full_filtered {} --> data_aig {}, data_air {}, tt {}, ff {}".format(len(self.data_full_filtered), len(self.data_aig), len(self.data_air), tt, ff))


def has_unknown_keyword(string):
    for key in unknown_keyword:
        if key in string:
            return True
    return False

unknown_list = ['unknown', 
                'Unknown',
                'not provided in the passage.', 
                'Unknown.',
                'unclear',
                'unknown.',
                'unclear.',
                'none',
                'None.',
                'None',
                'unknown.',
                'Unknown.']

unknown_keyword = ['does not mention',
                    'no information provided',
                    'no information',
                    'not provided', 
                    'no mention of',
                    'not provide',
                    'not mention',
                    'not enough information'
                    ]

prefix_example = {
    "Dataset": 'NQ',
    "LLMs":'GPT4',
    "Generator":'GPT4',
    "Reader":'GPT4'
}

def get_key_from_dict(d):
    key = ''
    for i,(k,v) in enumerate(d.items()) :
        key = key+k+'-'+v+'_'
    key = key[:-1]
    return key