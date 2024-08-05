from .llm import llm
from .jsonl import dump_jsonl, load_all_jsonl
from .data import find_with_question
from .evaluation import has_answer
from tqdm import tqdm
import logging
import os
import concurrent.futures
import threading
def rmreturn(s):
    s = s.replace('\n\n', ' ')
    s = s.replace('\n', ' ')
    return s.strip()

class Gen:
    def __init__(self, prompt_template, output_file, api_args, process_num = None) -> None:
        # init api: api_args['api_key']是list
        if process_num == None or process_num==0:
            self.process_num = len(api_args['api_key']) 
        else:
            self.process_num = process_num
        print("Use server num: {}".format(self.process_num))
        self.llm_server = []
        for i in range(self.process_num):
            self.llm_server.append(llm(api_args['engine'], api_args['api_key'][i], api_args['temperature'], api_args['max_tokens']))
        print("api_args:{}".format(api_args))
        # init prompt
        self.prompt_template = prompt_template
        self.output_file = output_file
        self.lock = threading.Lock()
    
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
        with self.lock:
            dump_jsonl(item, self.output_file)

    def _request(self, llm_index, prompt):
        return {"index": llm_index, "response": self.llm_server[llm_index].request(prompt), "prompt": prompt}
    
    def cocurrent_request(self, prompts):
        assert len(prompts) <= self.process_num
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.process_num) as executor:
            task = []
            for k, p in enumerate(prompts):
                future = executor.submit(self._request, k, p)
                task.append(future)
            results = []
            for future in concurrent.futures.as_completed(task):  # 并发执行
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print('generated an exception:{}'.format(exc))

        results.sort(key=lambda x: x['index'], reverse=False)
        generated_text = []
        for k, r in enumerate(results):
            assert r['prompt'] == prompts[k]
            generated_text.append(r['response'])
        return generated_text
        
    def create_prompt(self, item):
        assert 'question' in item
        return self.prompt_template.format(query=item['question'])
    
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
        Gen模型的prompt和response长度都为1
        '''
        data = self.check_exist(data)
        if self.process_num ==1:
            saved_data=[]
            for item in tqdm(data):
                prompt = self.create_prompt(item)
                item['prompt'] = [prompt]
                response = self.llm_server[0].request(prompt)
                item['response'] = [response]

                self.save_item(item)
                saved_data.append(item)
        else:
            saved_data=[]
            pbar = tqdm(total=len(data))
            index = 0
            pbar.update(index)
            while index<len(data):
                batch_item = []
                batch_prompt = []
                for _ in range(self.process_num):
                    if index >= len(data):
                        break
                    item = data[index]
                    index+=1
                    prompt = self.create_prompt(item)
                    item['prompt'] = prompt
                    batch_prompt.append(prompt)
                    batch_item.append(item)

                # batch request
                logging.info("-------------index: {}------------".format(index))
                logging.info(batch_prompt)
                responses = self.cocurrent_request(batch_prompt)
                logging.info(responses)
                for k, item in enumerate(batch_item):
                    item['response'] = responses[k]
                    # save
                    self.save_item(item)
                    saved_data.append(item)
                pbar.update(len(batch_prompt))
        return saved_data
    
    def _forward_single_server(self, data, server_index):
        saved_data=[]
        for item in tqdm(data):
            prompt = self.create_prompt(item)
            item['prompt'] = [prompt]
            response = self.llm_server[server_index].request(prompt)
            item['response'] = [response]
            if server_index == 0:
                logging.info("\nPrompt:\n{} \n Response: \n{}\n".format(prompt, response))
            self.save_item(item)
            saved_data.append(item)
        return saved_data
    
    def split_data_for_multi_thread(self,data):
        data_len = len(data)
        interval = data_len // self.process_num
        beg = [i * interval for i in range(self.process_num)]
        end = [(i + 1) * interval for i in range(self.process_num)]
        end[-1] = data_len
        print("slices：[beg] [end]")
        print(beg)
        print(end)
        return beg, end

    def forward_multi_thread(self, data):
        data = self.check_exist(data=data)
        beg, end = self.split_data_for_multi_thread(data=data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.process_num) as executor:
            task = []
            for server_index in range(self.process_num):
                future = executor.submit(self._forward_single_server, data[beg[server_index]:end[server_index]], server_index)
                task.append(future)
            results = []
            for future in concurrent.futures.as_completed(task):
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as exc:
                    print('generated an exception:{}'.format(exc))
        return results


class Read(Gen):
    def __init__(self, prompt_template, output_file, api_args, process_num = None) -> None:
        super().__init__( prompt_template, output_file, api_args, process_num=process_num)
    
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
