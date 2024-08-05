import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
from rouge import Rouge
import re
import math

class SimpleTokenizer(object):
	ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
	NON_WS = r'[^\p{Z}\p{C}]'

	def __init__(self):
		"""
		Args:
			annotators: None or empty set (only tokenizes).
		"""
		self._regexp = regex.compile(
			'(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
			flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
		)

	def tokenize(self, text, uncased=False):
		matches = [m for m in self._regexp.finditer(text)]
		if uncased:
			tokens = [m.group().lower() for m in matches]
		else:
			tokens = [m.group() for m in matches]
		return tokens


def check_answer(example, tokenizer) -> List[bool]:
	"""Search through all the top docs to see if they have any of the answers."""
	answers = example['answers']
	ctxs = example['ctxs']

	hits = []

	for _, doc in enumerate(ctxs):
		text = doc['text']

		if text is None:  # cannot find the document for some reason
			hits.append(False)
			continue

		hits.append(has_answer(answers, text, tokenizer))

	return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
	"""Check if a document contains an answer string."""
	text = _normalize(text)
	text = tokenizer.tokenize(text, uncased=True)

	for answer in answers:
		answer = _normalize(answer)
		answer = tokenizer.tokenize(answer, uncased=True)
		for i in range(0, len(text) - len(answer) + 1):
			if answer == text[i: i + len(answer)]:
				return True
	return False


def _normalize(text):
	return unicodedata.normalize('NFD', text)


def normalize_answer(s):
	def remove_articles(text):
		return regex.sub(r'\b(a|an|the)\b', ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
	return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
	return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
	prediction_tokens = normalize_answer(prediction).split()
	ground_truth_tokens = normalize_answer(ground_truth).split()
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1


def f1(prediction, ground_truths):
	return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
	rouge = Rouge()
	# no normalization
	try:
		scores = rouge.get_scores(prediction, ground_truth, avg=True)
	except ValueError:  # "Hypothesis is empty."
		return 0.0
	return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
	return max([rougel_score(prediction, gt) for gt in ground_truths])


# copy from https://github.com/wyu97/GenRead/blob/main/evaluation.py
## RenGen更改以适应新的数据形式
def eval_recall(infile, key):

	tokenizer = SimpleTokenizer()
	lines = infile

	has_answer_count = 0
	answer_lengths = []
	for line in lines:
		answer = line['answer']
		output = ' || '.join(line[key])

		if has_answer(answer, output, tokenizer):
			has_answer_count += 1

		answer_lengths.append(len(output.split()))

	recall = round(has_answer_count/len(lines), 4)
	lens = round(np.mean(answer_lengths), 4)

	return recall, lens


def eval_question_answering(infile):

	lines = open(infile, 'r').readlines()[1:]

	exact_match_count = 0
	answer_lengths = []
	for line in lines:
		line = json.loads(line)
		answer = line['answer']
		output = line['output'][0]

		if ems(output, answer): # EM evaluation
			exact_match_count += 1
		
		answer_lengths.append(len(output.split()))

	em = round(exact_match_count/len(lines), 4)
	lens = round(np.mean(answer_lengths), 4)

	return em, lens


def eval_fact_checking(infile):

	tokenizer = SimpleTokenizer()
	lines = open(infile, 'r').readlines()[1:]

	exact_match_count = 0
	answer_lengths = []
	for line in lines:
		line = json.loads(line)
		answer = line['answer']
		output = line['output'][0]

		if answer == ["refutes"]:
			answer = ["refutes", "no", "false"]
		if answer == ["supports"]:
			answer = ["supports", "yes", "true"]

		if has_answer(answer, output, tokenizer):
			exact_match_count += 1
		
		answer_lengths.append(len(output.split()))

	em = round(exact_match_count/len(lines), 4)
	lens = round(np.mean(answer_lengths), 4)

	return em, lens


def eval_dialogue_system(infile):

	lines = open(infile, 'r').readlines()[1:]

	f1_scores = []
	rl_scores = []
	answer_lengths = []
	for line in lines:
		line = json.loads(line)
		answer = line['answer']
		output = line['output'][0]

		f1_scores.append(f1(output, answer))
		rl_scores.append(rl(output, answer))
		answer_lengths.append(len(output.split()))

	F1 = round(np.mean(f1_scores), 4)
	RL = round(np.mean(rl_scores), 4)
	lens = round(np.mean(answer_lengths), 4)

	return F1, RL, lens


# from tevatron evalution
def recall_k(item, topk, regex=False):
	# item 'answer' 'response'
	tokenizer = SimpleTokenizer()
	accuracy = { k : [] for k in topk }
	max_k = max(topk)

	for qid in range(len(item)):
		answers = item[qid]['answer']
		contexts = item[qid]['response']
		has_ans_idx = max_k  # first index in contexts that has answers

		for idx, ctx in enumerate(contexts):
			if idx >= max_k:
				break
			if 'has_answer' in ctx:
				if ctx['has_answer']:
					has_ans_idx = idx
					break
			else:
				text = ctx
				if has_answer(answers, text, tokenizer):
					has_ans_idx = idx
					break

		for k in topk:
			accuracy[k].append(0 if has_ans_idx >= k else 1)

	for k in topk:
		accuracy[k] = np.mean(accuracy[k])
	return accuracy

# jaccard
tokenizer = SimpleTokenizer()
def cut_sentences(content):
	sentences = re.split(r'(\.|\!|\?|。|！|？|\.{6})', content)
	result = []
	for sen in sentences:
		if len(sen)>1:
			result.append(sen)
	return result

def jaccard_similarity(sen1, sen2):
	sen1_set = set(tokenizer.tokenize(sen1))
	sen2_set = set(tokenizer.tokenize(sen2))
	inter = sen1_set.intersection(sen2_set)
	return len(inter)/(len(sen1_set) + len(sen2_set) - len(inter))

def document_question_jaccard_similarity(document, question):
	sentences = cut_sentences(document)
	result = []
	for s in sentences:
		sim = jaccard_similarity(s, question)
		result.append(sim)
	return result

def max_jaccard_similarity(document, question, return_max_sentence=False):
	sentences = cut_sentences(document)
	max_jaccard = 0
	max_sentence = ''
	for s in sentences:
		sim = jaccard_similarity(s, question)
		if sim>max_jaccard:
			max_jaccard = sim
			max_sentence = s
	if return_max_sentence == True:
		return max_sentence
	return max_jaccard

# bertscore
from bert_score import score
def bertscore_similarity(cand, ref, device=0):
	P, R, F1 = score(cand, ref, lang='en', verbose=True, device=device)
	return F1

def document_question_bertscore_similarity(passage_list, question_list, device=0):
	#  input all questions and passages
	# output a [] for each passages
	flag = [0]
	cand, ref = [], []
	for k, passage in enumerate(passage_list):
		sentences = cut_sentences(passage)
		cand.extend(sentences)
		for _ in sentences:
			ref.append(question_list[k])
		flag.append(len(cand))
	sim = bertscore_similarity(cand, ref, device).tolist()
	result = []
	for k in range(len(question_list)):
		result.append( sim[flag[k]:flag[k+1]])
	return result

def max_bertscore_similarity(passage_list, question_list, return_max_sentence=False):
	flag = [0]
	cand, ref = [], []
	for k, passage in enumerate(passage_list):

		sentences = cut_sentences(passage)
		cand.extend(sentences)
		for _ in sentences:
			ref.append(question_list[k])
		flag.append(len(cand))
	sim = bertscore_similarity(cand, ref).tolist()
	max_sim = []
	for k in range(len(question_list)):
		if sim[flag[k]:flag[k+1]] != []:
			max_sim.append(max( sim[flag[k]:flag[k+1]] ))
		else:
			max_sim.append(0)
			print(f"warning! find a empty sentence: {question_list[k]}")
	return max_sim

def compute_ppl_from_logprobs(logprobs:list):
	return math.exp(-sum(logprobs))


import spacy
from tqdm import tqdm
def remove_repititive_instruction_in_generated_passage(passage):
	'''
	Some articles generated by llama2 have repetitive instruction situations, such as:  
	"
	Sure!  Here's a background document from Wikipedia to answer the question When was The Miraculous Journey of Edward Tulane   published?   
	"
 	Find the first \n\n and remove the contents before it.  Avoid repeated questions affecting the calculation of similarity. 
	'''
	# Find the index of the first occurrence of '\n\n'
	split_index = passage.find('\n\n')
	# print(split_index)
	# print(len(passage))
	
	# Check if '\n\n' is found and if it is beyond the 50% mark of the passage
	if split_index != -1 and split_index < len(passage) / 2:
		return passage[split_index+2:]
	else:
		# If '\n\n' is not found or it's before the 50% mark, return the original passage
		return passage

def get_jaccard_distribution(result, key, remove_repititive_instruction=False):    
	distribution = []
	for item in result:
		if isinstance(result[0][key], list):
			if remove_repititive_instruction == True:
				item[key][0] = remove_repititive_instruction_in_generated_passage(item[key][0])
			distribution.append(max(document_question_jaccard_similarity(item[key][0], item['question'])))
		else:
			if remove_repititive_instruction == True:
				item[key] = remove_repititive_instruction_in_generated_passage(item[key])
			distribution.append(max(document_question_jaccard_similarity(item[key], item['question'])))
	return distribution

def get_bertscore_distribution(result, key, remove_repititive_instruction=False):
	distribution = []
	passage_list, question_list = [], []
	for item in result:
		if isinstance(result[0][key], list):
			passage_list.append(item[key][0])
		else:
			passage_list.append(item[key])
		question_list.append(item['question'])
	if remove_repititive_instruction==True:
		removed_passage_list = [remove_repititive_instruction_in_generated_passage(i) for i in passage_list]
		passage_list = removed_passage_list
	distribution = max_bertscore_similarity(passage_list,question_list)
	return distribution

def get_length_distribution(result, key):
	if isinstance(result[0][key], list):
		passages = [t[key][0] for t in result]
	else:
		passages = [t[key] for t in result]
	length_distribution = spacy_length(passages)
	return length_distribution
	
def spacy_length(passage_list):
	nlp = spacy.load("en_core_web_sm")
	words_len = []
	for doc in tqdm(nlp.pipe(passage_list, disable=["parser", "ner", "lemmatizer"], n_process=1)):
		tokens = [i for i in nlp(doc) if not (i.is_space or i.is_punct)]
		words_len.append(len(tokens))
	return  words_len