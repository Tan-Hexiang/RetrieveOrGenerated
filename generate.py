import argparse
import os
import logging

from src.jsonl import load_all_jsonl, config_log, dump_jsonl
from src.evaluation import  eval_recall
from src.model import Gen
from src.data import find_with_question
from src.key import used_keys
from src.model_llama import Gen_llama
llama_path = [
    'llama_model/7b-chat',
    'llama_model/13b-chat',
    'llama_model/7b-chat-hf',
    'llama_model/13b-chat-hf'
]

api_args = {
    'engine':'',
    'api_key':used_keys,
    'temperature':0,
    'max_tokens':300
}
def evaluate_recall(result):
    recall, len = eval_recall(result, 'response')

    return {
        "recall": recall,
        "length": len
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
    )
    parser.add_argument("--type", default=None, type=str, required=True,
        help="type: [Gen, RAGen, RAGen_full, RAGen_completion], should be either 1 or 2",
    )
    parser.add_argument("--split", default=None, type=str, required=True,
        help="dataset split: [train, dev, test]",
    )
    parser.add_argument("--engine", default='text-davinci-002', type=str, required=False,
        help="text-davinci-002 (used in our experiments), code-davinci-002",
    )
    parser.add_argument("--ctxs_file", type=str, default='none', required=False)
    parser.add_argument('--ctxs_key', type=str, default='none', required=False)
    parser.add_argument('--ctxs_num', type=int, default=1, required=False)
    parser.add_argument("--decoding", default='greedy', type=str, required=False, help='[greedy, sample], affect the temperature')
    parser.add_argument("--pid", default=1, type=int)
    parser.add_argument('--process_num', type=int, default=0)
    # only used when using local model like llama
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--device_map", type=str, default='sequential')
    parser.add_argument("--debug_mode", action='store_true')
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
    api_args['max_tokens'] = 300
    if args.decoding == 'greedy':
        api_args['temperature'] = 0
    else:
        raise Exception("Do not support other sample method")
    
    # output path
    outputfolder = f'Generated-context-{args.decoding}-{args.engine}/{args.dataset}-{args.split}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.type}-with-{args.ctxs_num}-{args.ctxs_key}-p{args.pid}.jsonl'
    metricfile = f'{outputfolder}/Recal@k.jsonl'
    config_log(outputfolder, f'{args.type}-with-{args.ctxs_num}-{args.ctxs_key}-p{args.pid}')
    
    # get prompt_template
    prompt_list = load_all_jsonl('source/prompt.jsonl')
    prompt_template = None
    for p in prompt_list:
        if p['task'] == datatype and p['type'] == args.type and p['pid'] == args.pid:
            prompt_template = p['prompt_template']
            break
    if prompt_template==None: raise Exception("fail to find prompt template")
    # qa data
    inputfile = f'indatasets/{args.dataset}/{args.dataset}-{args.split}.jsonl'
    qa_data = load_all_jsonl(inputfile)

    # add ctxs
    if args.type in ['RAGen', 'RAGen_all', 'RAGen_completion']:
        ctxsfile = args.ctxs_file
        assert ctxsfile != 'none'
        ctxsdata = load_all_jsonl(ctxsfile)
    else:
        print("ctxs is none")
        ctxsdata = None

    data = []
    for k, qa in enumerate(qa_data):
        item={
            "question": qa['question'],
            "answer": qa['answer'],
        }
        if ctxsdata != None:
            # assert ctxsdata[k]['reference'] == item['answer']
            ctxs_item = find_with_question(qa['question'], ctxsdata)
            if ctxs_item == None: raise Exception("Can not find questin {}".format(qa['question']))
            item['ctxs'] = ctxs_item[args.ctxs_key][:args.ctxs_num]
        data.append(item)
    logging.info ("Total data to process {}".format(len(data)))
    logging.info("\n Data[0]:{}".format(data[0]))
    
    # debug model
    if args.debug_mode:
        data = data[:100]
        logging.error("Start debug mode, only process 100 examples")

    # debug 
    debug_info = {
        "dataset": args.dataset,
        "args": vars(args),
        "outputfile": outputfile,
        "prompt_template": prompt_template
    }
    logging.info(debug_info)
    
    # prepare llama model
    if args.engine in llama_path:
        logging.info('Using llama model in {}'.format(args.engine))
        if args.type == 'Gen':
            llm = Gen_llama(prompt_template, outputfile, api_args, batch_size=args.batch_size, device_map=args.device_map)
        else: 
            raise Exception("Unexpected type {}".format(args.type))
        # llama batch request
        result = llm.forward(data)
    # API request
    else:
        # create llm
        if args.type == 'Gen':
            llm = Gen(prompt_template, outputfile, api_args, process_num=args.process_num)
        else:
            raise Exception("Unexpected type {}".format(args.type))
        # multi process request
        result = llm.forward_multi_thread(data)

    # read the whole result
    if len(result) != len(data):
        result = load_all_jsonl(outputfile)
    # evaluation
    recall_dict = evaluate_recall(result)
    metric_file = f'{outputfolder}/metric.jsonl'
    metric = {
        "recall": recall_dict,
        "dataset": args.dataset,
        "args": vars(args),
        "outputfile": outputfile,
        "prompt_template": prompt_template
    }
    logging.info(metric)
    dump_jsonl(metric, metric_file)





    