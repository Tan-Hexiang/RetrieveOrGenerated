import logging
from tqdm import tqdm
from .llm import llm, llama_llm
from .jsonl import dump_jsonl, load_all_jsonl, dump_all_jsonl
from .data import find_with_question
from .evaluation import has_answer
import torch
import os

class Gen_llama:
    def __init__(self, prompt_template, output_file, api_args, batch_size, device_map, system_instruction=None, logprobs=False) -> None:
        self.logprobs = logprobs
        self.model = llama_llm(model_path=api_args['engine'], device_map=device_map, system_instruction=system_instruction, logprobs=logprobs)
        self.batch_size = batch_size
        self.max_tokens = api_args['max_tokens']
        # init prompt
        self.prompt_template = prompt_template
        self.output_file = output_file
    
    def check_exist(self, data):
        if os.path.exists(self.output_file):
            exist_data = load_all_jsonl(self.output_file)
            flag = []
            for k, example in enumerate(data):
                if find_with_question(example['question'], exist_data) == None:
                    flag.append(k)
            new_data = []
            for i in flag:
                new_data.append(data[i])
            logging.info("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), len(exist_data), len(new_data)))
            print("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), len(exist_data), len(new_data)))
            return new_data        
        else:
            logging.info("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), 0, len(data)))
            print("Total data: {}, Exist data: {}, Data to proess: {}".format(len(data), 0, len(data)))
            return data

    def save_item(self, item):
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            dump_jsonl(item, self.output_file)

    def save_items(self, items):
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            dump_all_jsonl(items, self.output_file, append=True)
        
    def create_prompt(self, item):
        assert 'question' in item
        return self.prompt_template.format(query=item['question'])
    
    def create_prompts(self, items):
        assert 'question' in items[0]
        prompts = []
        for item in items:
            prompts.append(self.prompt_template.format(query=item['question']))
        return prompts
    
    def forward(self, data):
        '''
        data format
        {
        # input
        "question": "who is the president of usa right now?",
        "answer": ["Donald Trump"],
        "ctxs": ["",],
        # output
        "prompt": ["",],
        "response": [""],
        }
        '''
        data = self.check_exist(data)

        saved_data=[]
        batchs = [data[x:x+self.batch_size] for x in range(0, len(data), self.batch_size)]
        for batch in tqdm(batchs):
            prompts = self.create_prompts(batch)
            if self.logprobs ==True:
                responses, logprobs_list, tokens_list = self.model.request(batch_prompt=prompts, max_tokens=self.max_tokens)
            else:
                responses = self.model.request(batch_prompt=prompts, max_tokens=self.max_tokens)
            # save result
            for k, _ in enumerate(batch):
                batch[k]['prompt'] = prompts[k]           
                batch[k]['response'] = [responses[k]]
                if self.logprobs ==True:
                    batch[k]['logprobs'] = logprobs_list[k]
                    batch[k]['tokens'] = tokens_list[k]
            saved_data.extend(batch)
            # save to disk
            self.save_items(batch)
        return saved_data

class Read_llama(Gen_llama):
    def __init__(self, prompt_template, output_file, api_args, batch_size, device_map, logprobs=False) -> None:
        # Ensure output format
        self.system_instruction = "Your response must strictly adhere to the format ':%d,' where %d is a single word or phrase. The reply must begin with a colon and contain no additional verbosity. An example of a good response: ':Joe Biden'"
        super().__init__(prompt_template, output_file, api_args, batch_size, device_map, system_instruction=self.system_instruction,logprobs=logprobs)

    def create_prompt(self, item):
        assert 'question', 'ctxs' in item
        def rmreturn(s):
            s = s.replace('\n\n', ' ')
            s = s.replace('\n', ' ')
            return s.strip()
        if item['ctxs'] == []:
            prompt = self.prompt_template.format(query=item['question'])
        else:
            backinfo=''
            for ctxs in item['ctxs']:
                backinfo = backinfo+ '\n'+ rmreturn(ctxs)
            prompt = self.prompt_template.format(query=item['question'], background=backinfo)
        return prompt

    def create_prompts(self, items):
        assert 'question', 'ctxs' in items[0]
        def rmreturn(s):
            s = s.replace('\n\n', ' ')
            s = s.replace('\n', ' ')
            return s.strip()
        prompts = []
        for item in items:
            if item['ctxs'] == []:
                prompts.append(self.prompt_template.format(query=item['question']))
            else:
                backinfo=''
                for ctxs in item['ctxs']:
                    backinfo = backinfo+ '\n'+ rmreturn(ctxs)
                prompts.append(self.prompt_template.format(query=item['question'], background=backinfo))
        return prompts
    