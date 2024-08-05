
# a example scripts
# python find_gen_ctx_length_similar_to_ret_cxt.py --all_gen_dir Generated-context-greedy-llama_model/7b-chat/nq-test --ret_file backgrounds-retrieval/nq-test/retrieval_result.jsonl

# python find_gen_ctx_length_similar_to_ret_cxt.py --all_gen_dir Generated-context-greedy-llama_model/7b-chat/nq-test --ret_file backgrounds-retrieval/nq-test/retrieval_result.jsonl

import argparse
from src.jsonl import load_all_jsonl, dump_all_jsonl, dump_jsonl, config_log
from src.data import find_with_question
import logging
from pathlib import Path
from tqdm import tqdm
from src.evaluation import eval_recall
import spacy
def add_word_length_key(data, is_ret = False):
    if is_ret:
        passage_list = [i['contriever'][0] for i in data]
    else:
        passage_list = [i['response'][0] for i in data]
    nlp = spacy.load("en_core_web_sm")
    words_len = []
    # , n_process=8, batch_size=200
    for doc in tqdm(nlp.pipe(passage_list, disable=["parser", "ner", "lemmatizer"], n_process=8)):
        tokens = [i for i in nlp(doc) if not (i.is_space or i.is_punct)]
        words_len.append(len(tokens))
    # save length to key
    for k, _ in enumerate(data):
        data[k]['word_length'] = words_len[k]
    mean_length = sum(words_len)/len(words_len)
    return data, mean_length, words_len

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--all_gen_dir", default=None, type=str, required=True,
        help="dir contain all gen file, we exclude file with p1",
    )
    parser.add_argument("--ret_file", default=None, type=str)
    parser.add_argument("--only_plot", action="store_true")
    args = parser.parse_args()
    # output path
    output_path = args.all_gen_dir+'/gen_with_similar_length.jsonl'
    distribution_path = args.all_gen_dir+'/gen_with_similar_length.pdf'
    # log
    config_log(args.all_gen_dir, 'gen_prompt_similar_length.log')
    # find all files
    gen_dir = Path(args.all_gen_dir)
    gen_files = []
    for file in list(gen_dir.iterdir()):
        # filter p1
        if 'p1.json' not in str(file):
            # save jsonl
            if file.suffix == '.jsonl':
                # metric.jsonl
                if 'metric' not in str(file):
                    # remove RAGen
                    if 'RAGen' not in str(file):
                        gen_files.append(str(file))
    logging.error("All gen ctxs found: {}".format(gen_files))
    logging.error("Using ret ctxs in: {}".format(args.ret_file))
    # load data
    ret_data = load_all_jsonl(args.ret_file)
    gen_datas = []
    for file in gen_files:
        gen_datas.append(load_all_jsonl(file))

    # compute word length to key 'word_length'
    ret_data, ret_mean_length, ret_word_length_list = add_word_length_key(ret_data, is_ret=True)
    logging.error("Ret mean length : {}".format(ret_mean_length))
    if args.only_plot !=True:
        for k, _ in enumerate(gen_datas):
            gen_datas[k], mean_length, _ = add_word_length_key(gen_datas[k])
            logging.error("gen {} mean length : {}".format(k, mean_length))
        # find the example has most similar length with ret ctxs
        result = []
        for example in tqdm(ret_data):
            question = example['question']
            ret_length = example['word_length']
            min_delta = 1000
            final_item = None
            for i, _ in enumerate(gen_datas):
                try:
                    gen_item = find_with_question(question,gen_datas[i])
                except:
                    print("None item for question: {}".format(question))
                if abs(gen_item['word_length']-ret_length) < min_delta:
                    final_item = gen_item
                    min_delta = abs(gen_item['word_length']-ret_length)
            # save item with similar length
            result.append(final_item)
        dump_all_jsonl(result, output_path)
    else:
        result = load_all_jsonl(output_path)
    _, result_mean_length, result_word_length_list = add_word_length_key(result)
    logging.error('ret mean length: {}, result mean length: {}'.format(ret_mean_length, result_mean_length))
    
    # Plot distribution
    import matplotlib.pyplot as plt
    import numpy as np
    plt.hist(result_word_length_list, alpha=0.5,bins=np.arange(0, 200, step=1), label='Generated Context')
    plt.hist(ret_word_length_list, alpha=0.5,bins=np.arange(0, 200, step=1), label='Retrieved Context')
    plt.legend(loc='upper right')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    # plt.title("Length Distribution")
    plt.savefig(distribution_path, dpi=600, facecolor='w', edgecolor='w',orientation='portrait', format='pdf', transparent=False)
    # plt.show()     
    for k, data in enumerate(gen_datas):
        recall,_ = eval_recall(gen_datas[k], 'response')
        logging.error("gen {} recall: {}".format(k,recall))
    for k, _ in enumerate(ret_data):
        ret_data[k]['contriever'] = ret_data[k]['contriever'][:1]
    ret_recall,_ = eval_recall(ret_data, 'contriever')
    logging.error("ret recall: {}".format(ret_recall))
    # result recall
    result_recall,_   = eval_recall(result, 'response')
    logging.error("result recall: {}".format(result_recall))
        

