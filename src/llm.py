import openai
import time
import logging
import os
from .jsonl import dump_jsonl, load_all_jsonl
from .data import find_with_question
from llama import Llama, Dialog
import torch
from tqdm import tqdm


gpt = [
    'text-davinci-003',
    'text-davinci-002'
]

gpt_chat =[
    'gpt-3.5-turbo-0613',
    'gpt-4-0613',
    'gpt-4-1106-preview'
]

llama = [
    'llama-2-7b',
    'llama-2-13b',
    'llama-2-70b',
]

llama_chat = [
    'llama-2-7b-chat',
    'llama-2-13b-chat',
    'llama-2-70b-chat'
]

class llm:
    def __init__(self, model, api_key, temperature, max_tokens) -> None:
        openai.api_key = api_key
        assert (model in gpt) or (model in gpt_chat) or (model in llama) or (model in llama_chat), "model {} not in the available model list".format(model)
        self.model = model
        self.temp = temperature
        self.max_tokens = max_tokens
        if (self.model in llama) or (self.model in llama_chat):
            self.generator = self.prepare_llama()
    
    def prepare_llama(self):
        generator = Llama.build(
            ckpt_dir=self.model,
            tokenizer_path='tokenizer.model',
            max_seq_len=2048,
            max_batch_size=1,
            model_parallel_size=1
        )
        return generator

    def request_api(self, prompt):
        if self.model in gpt:
            response = openai.Completion.create(
                engine=self.model,
                prompt=prompt,
                temperature=self.temp,
                max_tokens=self.max_tokens
            )
            generated_text = response.choices[0].text
            # print("use completion")
        elif self.model in gpt_chat:
            # print("use chat completion")
            completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
                model = self.model,
                temperature = self.temp,
                max_tokens = self.max_tokens,
                messages = [ # Change the prompt parameter to the messages parameter
                    {'role': 'user', 'content': prompt}
                ],
            )
            generated_text = completion['choices'][0]['message']['content']

        elif self.model in llama:
            results = self.generator.text_completion(
                [prompt],
                max_gen_len=self.max_tokens,
                temperature=self.temp,
            )
            generated_text = results[0]['generation']

        elif self.model in llama_chat:
            dialogs =[
            [{"role": "user", "content": prompt}]
            ]
            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=self.max_tokens,
                temperature=self.temp,
            )
            generated_text = results
            for dialog, result in zip(dialogs, results):
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n")
                generated_text = results[0]['generation']['content']
        else:
            raise Exception("Unexcepted model {}".format(self.model))
        if generated_text == None or generated_text == [None]:
            raise Exception("None output")
        return generated_text

    def request(self, prompt):
        while True:
            try:
                res = self.request_api(prompt=prompt)
                # logging.info("Prompt:{}".format(prompt))
                # logging.info("Response:{}".format(res))
                break
            except openai.error.RateLimitError as e:
                print('\nRateLimitError\t', e, '\tRetrying...')
                time.sleep(2)
            except openai.error.ServiceUnavailableError as e:
                print('\nServiceUnavailableError\t', e, '\tRetrying...')
                # time.sleep(1)
            except openai.error.Timeout as e:
                print('\nTimeout\t', e, '\tRetrying...')
                # time.sleep(1)
            except openai.error.APIError as e:
                print('\nAPIError\t', e, '\tRetrying...')
                # time.sleep(1)
            except openai.error.APIConnectionError as e:
                print('\nAPIConnectionError\t', e, '\tRetrying...')
                # time.sleep(1)
            except Exception as e:
                print(e)
                res = None
                break
        return res

    

class llama_llm:
    def __init__(self, model_path, device_map='sequential', load_in_8bit=False, system_instruction = None, logprobs=False) -> None:
        self.generator = Llama.build(
            ckpt_dir=model_path,
            tokenizer_path=model_path+'/tokenizer.model',
            max_seq_len=4096,
            max_batch_size=2,
        )
        self.logprobs = logprobs
        self.temperature = 0
        self.top_p = 0
        self.system_instruction = system_instruction
        logging.info("system instruction: {}".format(self.system_instruction))
        print("system instruction: {}".format(self.system_instruction))

    def process_prompt(self, prompts):
        dialogs: list[Dialog] = []
        # {"role": "user", "content": "what is the recipe of mayonnaise?"}]
        for prompt in prompts:
            if self.system_instruction == None:
                dialog = [{"role": "user", "content": prompt}]
            else:
                dialog = [
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt}
                ]
            dialogs.append(dialog)
        return dialogs

    def rm_prompt_in_response(self, batch_prompt, responses):
        # llama 2 will include the input in their output, rm it
        new_response = []
        for k,prompt in enumerate(batch_prompt):
            if prompt in responses[k]:
                new_response.append(responses[k].replace(prompt, ''))
        return new_response
    
    def request(self, batch_prompt, max_tokens):
        dialogs = self.process_prompt(batch_prompt)
        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=self.logprobs
        )
        logprobs_list = []
        tokens_list = []
        responses = []
        for r in results:
            responses.append(r['generation']['content'])
            try:
                if self.logprobs ==True:
                    tokens_list.append(r['tokens'])
                    logprobs_list.append(r['logprobs'])
            except:
                print("self.logprobs:{}".format(self.logprobs))
                print(results)
                print(tokens_list, logprobs_list)
        # responses = self.rm_prompt_in_response(batch_prompt, responses)
        # logging.info(batch_prompt, responses, '\n')
        if self.logprobs ==False:
            return responses
        else:
            return responses, logprobs_list, tokens_list
        


def check_exist(data, exist_data_path):
        if os.path.exists(exist_data_path):
            exist_data = load_all_jsonl(exist_data_path)
            flag = []
            for k, example in enumerate(data):
                if find_with_question(example['question'], exist_data) == None:
                    flag.append(k)
                    print(example['question'])
                else:
                    print(f"{example['question']==find_with_question(example['question'], exist_data)['question']}    {example['question']}    {find_with_question(example['question'], exist_data)['question']}")
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
    