import argparse
import os
import logging
import numpy as np
import random

from src.jsonl import load_all_jsonl, config_log, dump_jsonl
from src.evaluation import ems, f1, has_answer, SimpleTokenizer, recall_k, eval_recall
from src.model import Read
from src.data import find_with_question, truncation_with_length
from src.key import used_keys
from src.model_llama import Read_llama
from tqdm import tqdm
random.seed(4)
api_args = {
    'engine':'',
    'api_key': used_keys,
    'temperature':0,
    'max_tokens':300
}

llama_path = [
    'llama_model/7b-chat',
    'llama_model/13b-chat',
    'llama_model/7b-chat-hf',
    'llama_model/13b-chat-hf'
]
def evaluate_recall(result):
    # 计算输入ctxs的recall@k
    recall, len = eval_recall(result, 'ctxs')
    return recall

def evaluate_answer(result):
    mean_f1, mean_em = [], []
    for example in result:
        try:
            mean_f1.append(f1(example['response'][0], ground_truths=example['answer']))
            mean_em.append(ems(prediction=example['response'][0], ground_truths=example['answer']))
        except BaseException as e:
            print("error at question: {}".format(example['question']))
            print("Detail : {}".format(e))
    mean_em = sum(mean_em)/len(result)
    mean_f1 = sum(mean_f1)/len(result)
    return {
        "f1": mean_f1,
        "em": mean_em
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # used to find the prompt template
    parser.add_argument("--dataset", default=None, type=str, required=True,
        help="dataset name: [nq, tqa]",
    )
    parser.add_argument("--pid", default=1, type=int)
    # retrieved passage
    parser.add_argument("--retrieved_name", type=str)
    parser.add_argument("--ctxs_file", type=str, default='none', required=False)
    parser.add_argument('--ctxs_key', type=str, default='none', required=False)
    parser.add_argument('--ctxs_num', type=int, default=1, required=False) 
    # generated passage
    parser.add_argument('--generated_name', type=str)
    parser.add_argument("--generated_file", type=str, default='none', help='default use key \'response\'')
    parser.add_argument("--generated_key", type=str, default='response')
    parser.add_argument('--generated_num', type=int, default=1)
    # 合并顺序
    parser.add_argument("--merge_order", type=str, default='generated_first', help='[generated_first, retrieved_first, random]')
    # about llm
    parser.add_argument("--engine", default='text-davinci-002', type=str, required=False,
        help="text-davinci-002 (used in our experiments), code-davinci-002",
    )
    parser.add_argument('--process_num', type=int, default=0)
    # parser.add_argument('--need_check', type=bool, default=False)
    # only used when using local model like llama
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--device_map", type=str, default='sequential')
    parser.add_argument("--logprobs", action='store_true')
    parser.add_argument("--subdir", type=str, default='default')
    parser.add_argument("--subset_name", type=str, default='none')
    parser.add_argument("--truncation_generated_with_token_length", type=int, default = 0)
    args = parser.parse_args()
    if args.dataset in ['nq', 'webq', 'tqa', 'twiki']:
        datatype = 'question answering'
    elif args.dataset in ['fever', 'fm2']:
        datatype = 'fact checking'
    elif args.dataset in ['wizard']: 
        datatype = 'dialogue system'
    else: 
        raise NotImplementedError
    # api config
    api_args['engine'] = args.engine
    if datatype == 'dialogue system':
            api_args['max_tokens'] = 50
    else: # QA and Fact ...
            api_args['max_tokens'] = 10
    api_args['temperature'] = 0
    
    # output path
    if args.subset_name == 'none':
        outputfolder = f'Answer-with-{args.engine}/{args.dataset}'
    else:
        outputfolder = f'Answer-with-{args.engine}/{args.subset_name}'
    if args.subdir == 'default':
        outputfolder = outputfolder
    else:
        outputfolder = outputfolder + f'/{args.subdir}'
    os.makedirs(outputfolder, exist_ok=True)
    if args.logprobs ==True:
        # save log
        outputfile = f'{outputfolder}/logprobs_Retrieved-{args.retrieved_name}-{args.ctxs_num}-Generated-{args.generated_name}-p{args.pid}-trunclen-{args.truncation_generated_with_token_length}.jsonl'
        config_log(outputfolder, f'logprobs_Retrieved-{args.retrieved_name}-{args.ctxs_num}-Generated-{args.generated_name}-p{args.pid}-trunclen-{args.truncation_generated_with_token_length}')
    else:
        outputfile = f'{outputfolder}/Retrieved-{args.retrieved_name}-{args.ctxs_num}-Generated-{args.generated_name}-p{args.pid}-trunclen-{args.truncation_generated_with_token_length}.jsonl'
        config_log(outputfolder, f'Retrieved-{args.retrieved_name}-{args.ctxs_num}-Generated-{args.generated_name}-p{args.pid}-trunclen-{args.truncation_generated_with_token_length}')
    
    # get prompt_template
    prompt_list = load_all_jsonl('source/prompt.jsonl')
    prompt_template = None
    for p in prompt_list:
        if p['task'] == datatype and p['type'] == 'Read' and p['pid'] == args.pid:
            prompt_template = p['prompt_template']
            break
    
    logging.info(vars(args))
    # --------------------------------#
    # perpare data
    assert args.generated_file != 'none' or args.ctxs_file != 'none', "Need at least one type of passages"
    data = []
    if args.generated_file != 'none':
        generated_data = load_all_jsonl(args.generated_file)
        logging.info("generated_data[10]:{}".format(generated_data[10]))
        logging.info("truncation length：{}".format(args.truncation_generated_with_token_length))
        for example in tqdm(generated_data):
            assert args.generated_num <= len(example[args.generated_key])
            item={
                "question": example['question'],
                "answer": example['answer'],
                "generated_passage": example[args.generated_key][:args.generated_num]
            }
            # truncation if given truncation_generated_with_token_length
            if args.truncation_generated_with_token_length !=0:
                for k, _ in enumerate(item['generated_passage']):
                    item['generated_passage'][k] = truncation_with_length(item['generated_passage'][k], args.truncation_generated_with_token_length)
            data.append(item)    
        # also has ctxs
        if args.retrieved_name != 'none':
            retrieved_data = load_all_jsonl(args.ctxs_file)
            for k, example in enumerate(data):
                # match ret and gen ctxs
                retrieved_item = find_with_question(example['question'], retrieved_data)
                data[k]['retrieved_passage'] = retrieved_item[args.ctxs_key][:args.ctxs_num]
                # merge contexts
                if args.merge_order == 'generated_first':
                    data[k]['ctxs'] = data[k]['generated_passage'] + data[k]['retrieved_passage']
                elif args.merge_order == 'retrieved_first':
                    data[k]['ctxs'] = data[k]['retrieved_passage'] + data[k]['generated_passage']
                elif args.merge_order == 'random':
                    data[k]['ctxs'] = data[k]['generated_passage'] + data[k]['retrieved_passage']
                    random.shuffle(data[k]['ctxs'])
                else:
                    raise Exception("Unexpercted merge_order {}".format(args.merge_order))
        # only generated ctxs
        else:
            for k,example in enumerate(data):
                data[k]['ctxs'] = data[k]['generated_passage']
                data[k]['retrieved_passage'] = []
    # only retrieved passage
    elif args.retrieved_name != 'none': 
        retrieved_data = load_all_jsonl(args.ctxs_file)
        for example in retrieved_data:
            item={
                "question": example['question'],
                "answer": example['answer'],
                "retrieved_passage": example[args.ctxs_key][:args.ctxs_num],
                "generated_passage": [],
                "ctxs": example[args.ctxs_key][:args.ctxs_num],
            }
            data.append(item)
    else:
        logging.info("No ctxs")
        retrieved_data = load_all_jsonl(args.ctxs_file)
        for example in retrieved_data:
            item={
                "question": example['question'],
                "answer": example['answer'],
                "retrieved_passage": [],
                "generated_passage": [],
                "ctxs": [],
            }
            data.append(item)
            
    logging.info("Data[0]: {}".format(data[0]))
    logging.info("Data to process {}".format(len(data)))
    logging.info("data[10]:{}".format(data[10]))
    # --------------------------------#
    # debug 
    debug_info = {
        "dataset": args.dataset,
        "args": vars(args),
        "outputfile": outputfile,
        "prompt_template": prompt_template
    }
    logging.info(debug_info)

    # llama request
    if args.engine in llama_path:
        logging.info('Using llama model in {}'.format(args.engine))
        llm = Read_llama(prompt_template, outputfile, api_args, args.batch_size, args.device_map, logprobs=args.logprobs)
        result = llm.forward(data)
    else:
        # create llm
        llm = Read(prompt_template=prompt_template, output_file=outputfile, api_args=api_args ,process_num=args.process_num)
        result = llm.forward_multi_thread(data)

    # read the whole result and analyze
    if len(result) != len(data):
        result = load_all_jsonl(outputfile)
    
    # evaluation
    metric_dict = evaluate_answer(result)
    metric_file = f'{outputfolder}/metric.jsonl'
    ctxs_recall_k = evaluate_recall(result)
    metric = {
        "metric": metric_dict,
        'recall': ctxs_recall_k,
        "dataset": args.dataset,
        "args": vars(args),
        "outputfile": outputfile,
        "prompt_template": prompt_template
    }
    dump_jsonl(metric, metric_file)
    
    logging.info(metric)
    print(metric)





    