def rm_last_mark(question:str):
    if question[-1] == '?':
        return question[:-1]
    else:
        return question

def find_with_question(question, data, answers=None):
    for item in data:
        if rm_last_mark(item['question']) == rm_last_mark(question):
            if answers == None:
                return item
            else:
                if answers == item['answer']:
                    return item
    # raise Exception("Cannot find question: {}".format(question))
    # print("Cannot find question: {}".format(question))
    return None

import spacy
def rm_before_content(tokens):
    content_flag = 0
    for i, t in enumerate(tokens):
        if t.text == 'Content':
            content_flag = i
            break
    return tokens[content_flag+1:]

def truncation_with_length(passage, length = 100):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(passage)
    tokens = [i for i in doc if not (i.is_space or i.is_punct)]
    tokens = tokens[:length]
    return doc[:tokens[-1].i].text

def truncation_complete_sentence(passage, beg_length = 100, end_length=120):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(passage)
    tokens = [i for i in doc if not (i.is_space or i.is_punct)]
    index = -1
    for token in tokens[beg_length:end_length]:
        if token.is_sent_start==True:
            index = token.i
            break
    return doc[:index].text

def truncation_with_same_length(gen_passage, ret_passage):
    nlp = spacy.load("en_core_web_sm")
    gen_doc = nlp(gen_passage)
    ret_doc = nlp(ret_passage)
    len_ret = len([i for i in ret_doc])
    return gen_doc[:len_ret].text

def truncation_with_same_length_complete_sentence(gen_passage, ret_passage):
    nlp = spacy.load("en_core_web_sm")
    gen_doc = nlp(gen_passage)
    ret_doc = nlp(ret_passage)
    len_ret = len([i for i in ret_doc])
    tokens = [i for i in gen_doc]
    
    
    return gen_doc[:len_ret].text