import requests
import numpy as np
import os
import csv
import pickle
import spacy
from . import trees

def fetch_wikis():
	with open('wiki_links.txt', 'r') as f:
		for line in f:
			link = line.strip()


def fetch_wiki(url):
	response = requests.get(url)
	soup = bs(response.content, 'html.parser')
	print(soup.get_text())
	return


def create_vector(word, word2vec, word_vector_size, silent=False):
	# if the word is missing from Glove, create some fake vector and store in glove!
	vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
	word2vec[word] = vector
	if (not silent):
		print("utils.py::create_vector => %s is missing" % word)
	return vector


def init_babi(fname):
	print("==> Loading test from %s" % fname)
	tasks = []
	task = None
	for i, line in enumerate(open(fname)):
		id = int(line[0:line.find(' ')])
		if id == 1:
			task = {"C": "", "Q": "", "A": ""}

		line = line.strip()
		line = line.replace('.', ' . ')
		line = line[line.find(' ') + 1:]
		if line.find('?') == -1:
			task["C"] += line
		else:
			idx = line.find('?')
			tmp = line[idx + 1:].split('\t')
			task["Q"] = line[:idx]
			task["A"] = tmp[1].strip()
			tasks.append(task.copy())

	return tasks


def init_babi_deploy(fname, query):
	task = {'C': "", 'Q': query}
	with open(fname, 'r') as f:
		for line in f:
			line = line.strip()
			line = line.replace('.', ' . ')
			task['C'] += line
			# task['C'] += " "
	tasks = []
	tasks.append(task.copy())
	print(tasks)
	return tasks


def get_babi_raw(id, test_id):
	babi_map = {
		"1": "qa1_single-supporting-fact",
		"2": "qa2_two-supporting-facts",
		"3": "qa3_three-supporting-facts",
		"4": "qa4_two-arg-relations",
		"5": "qa5_three-arg-relations",
		"6": "qa6_yes-no-questions",
		"7": "qa7_counting",
		"8": "qa8_lists-sets",
		"9": "qa9_simple-negation",
		"10": "qa10_indefinite-knowledge",
		"11": "qa11_basic-coreference",
		"12": "qa12_conjunction",
		"13": "qa13_compound-coreference",
		"14": "qa14_time-reasoning",
		"15": "qa15_basic-deduction",
		"16": "qa16_basic-induction",
		"17": "qa17_positional-reasoning",
		"18": "qa18_size-reasoning",
		"19": "qa19_path-finding",
		"20": "qa20_agents-motivations",
		"MCTest": "MCTest",
		"19changed": "19changed",
		"joint": "all_shuffled",
		"sh1": "../shuffled/qa1_single-supporting-fact",
		"sh2": "../shuffled/qa2_two-supporting-facts",
		"sh3": "../shuffled/qa3_three-supporting-facts",
		"sh4": "../shuffled/qa4_two-arg-relations",
		"sh5": "../shuffled/qa5_three-arg-relations",
		"sh6": "../shuffled/qa6_yes-no-questions",
		"sh7": "../shuffled/qa7_counting",
		"sh8": "../shuffled/qa8_lists-sets",
		"sh9": "../shuffled/qa9_simple-negation",
		"sh10": "../shuffled/qa10_indefinite-knowledge",
		"sh11": "../shuffled/qa11_basic-coreference",
		"sh12": "../shuffled/qa12_conjunction",
		"sh13": "../shuffled/qa13_compound-coreference",
		"sh14": "../shuffled/qa14_time-reasoning",
		"sh15": "../shuffled/qa15_basic-deduction",
		"sh16": "../shuffled/qa16_basic-induction",
		"sh17": "../shuffled/qa17_positional-reasoning",
		"sh18": "../shuffled/qa18_size-reasoning",
		"sh19": "../shuffled/qa19_path-finding",
		"sh20": "../shuffled/qa20_agents-motivations",
	}
	if (test_id == ""):
		test_id = id
	babi_name = babi_map[id]
	babi_test_name = babi_map[test_id]
	babi_train_raw = init_babi(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
	                                        'data/en/%s_train.txt' % babi_name))
	babi_test_raw = init_babi(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
	                                       'data/en/%s_test.txt' % babi_test_name))
	return babi_train_raw, babi_test_raw


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
	if not word in word2vec:
		create_vector(word, word2vec, word_vector_size, silent)
	# word2vec[word] = np.random.rand(1,50)
	if not word in vocab:
		next_index = len(vocab)
		vocab[word] = next_index
		ivocab[next_index] = word

	if to_return == "word2vec":
		return word2vec[word]
	elif to_return == "index":
		return vocab[word]
	elif to_return == "one_hot":
		raise Exception("to_return = 'one_hot' is not implemented yet")



def _load_glove(dim=50):


	if dim == 3:
		return load_glove_visualisation()

	glove = {}
	# path = "/Users/meshde/Mehmood/EruditeX/data/glove/glove.6B.50d.txt"
	path = os.path.join(
		os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'),
		             'glove'), 'glove.6B.%sd.txt' % dim)
	with open(path, 'r', encoding="utf8") as f:
		for line in f:
			l = line.split()
			glove[l[0]] = list(map(float, l[1:]))
	return glove

def load_glove(dim=50):
	# if you were using the old load_glove fn, continue to use as is, this wont affect you
	if dim == 3:
		return _load_glove(dim)
	else:
		glove = {}
		file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/cache/')
		if os.path.isfile(os.path.join(file_path, 'glove_{}.pkl'.format(str(dim)))):
			with open(os.path.join(file_path, 'glove_{}.pkl'.format(str(dim))), 'rb') as f:
				glove = pickle.load(f)
		else:
			glove = _load_glove(dim)
			with open(os.path.join(file_path, 'glove_{}.pkl'.format(str(dim))), 'wb') as f:
				pickle.dump(glove, f)
		return glove

def load_glove_visualisation(recreate=False):
	path = os.path.join(
		os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'),
		             'glove'), 'glove_visualisation.pkl')
	if os.path.exists(path) and not recreate:
		with open(path, 'rb') as f:
			glove = pickle.load(f)
			return glove
	glove = {}
	for x in ['welcome', 'to', 'my', 'house', 'home', 'where', 'is', 'are', 'you']:
		glove[x] = np.random.rand(1, 3)
	with open(path, 'wb') as f:
		pickle.dump(glove, f)
	return glove


def get_vector(word, glove, dim=50):
	try:
		ans = np.array(glove[word]).reshape((1, dim))
		return ans
	except:
		return np.random.rand(1, dim)


def get_list_sequence(sentence, glove, dim=50):
	result = []
	for word in sentence.split():
		result.append(get_vector(word, glove, dim))
	return result


def get_vector_sequence(sentence, glove, dim=50):
	list_sequence = get_list_sequence(sentence, glove, dim)
	result = np.array(list_sequence).reshape((-1, dim))
	return result


def get_norm(x):
	x = np.array(x)
	return np.sum(x * x)


def get_var_name(var, namespace):
	return [name for name in namespace if namespace[name] is var][0]


def load_dep_tags():
	path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'depTags.txt')
	dep_tags = []
	dep_tags_dict = {}
	with open(path, 'r') as file1:
		for line in file1:
			l = line.split(':')
			dep_tags.append(l[0])
	dep_tags_dict = dict([(w, i) for i, w in enumerate(dep_tags)])
	return dep_tags_dict


def get_depTags_sequence(sentence, dep_tags_dict, nlp):
	result = []
	# nlp=spacy.load('en')
	doc = nlp(sentence)
	for w in doc:
		key = w.dep_.split('||')[0]
		result.append(dep_tags_dict[key.upper()])
	return np.array(result)


def get_sent_details(sentence, glove, dep_tags_dict, nlp, wVec_size=50):
	result1 = get_vector_sequence(sentence, glove, wVec_size)
	result2 = get_depTags_sequence(sentence, dep_tags_dict, nlp)
	return result1, result2



def get_dtree(sentence, nlp=None, dim=50):
	if not nlp:
		nlp = spacy.load('en')
	doc = nlp(sentence)
	sents = [sent for sent in doc.sents]
	sent = sents[0]
	glove = load_glove(dim)
	return get_tree_node(sent.root, glove, dim)

def get_tree_node(node, glove, dim=50):
	return trees.dt_node(node, glove, [get_tree_node(child, glove, dim) for child in node.children], dim)

def get_dtne_tree(sentence, nlp=None, dim=50):
	if not nlp:
		nlp = spacy.load('en')
	doc = nlp(sentence)

	for ent in doc.ents:
		ent.merge()

	sents = [sent for sent in doc.sents]
	sent = sents[0]

	glove = load_glove(dim)
	return get_dtne_node(sent.root, glove, dim)

def get_dtne_node(node, glove, dim=50):
	return trees.dtne_node(node, glove, [get_dtne_node(child, glove, dim) for child in node.children], dim)

def get_sentence_from_doc(doc):
	sents = [sent for sent in doc.sents]
	sent = sents[0]
	return sent

def pad_vector_with_zeros(arr, pad_width):
	return np.lib.pad(arr, pad_width=(0,pad_width), mode='constant')

def pad_matrix_with_zeros(arr, pad_width):
	return np.lib.pad(arr, pad_width=((0,pad_width),(0,0)), mode='constant')

def print_token_details(sentence):
	nlp = spacy.load('en')
	doc = nlp(sentence)

	print("Token", "\t", "POS", "\t", "DEP", "\t", "Head", "\t", "DEP_CODE")

	for token in doc:
		print(token, "\t", token.pos_, "\t", token.dep_, "\t", token.head, "\t", token.dep)
	return

def get_ne_index(ent_type):
	if ent_type == 0:
		return 0
	if ent_type == 448:
		return 18
	return ent_type - 378 + 1

def _process_wikiqa_dataset(mode, max_sent_len=50):
	questions = []
	answers = []
	# file = ".\Dataset\WikiQACorpus\WikiQA-dev.tsv"
	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/wikiqa/WikiQA-{}.tsv'.format(mode))
	qfile_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/cache/wikiqa_Q_{}.pkl'.format(mode))
	afile_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/cache/wikiqa_A_{}.pkl'.format(mode))
	
	if os.path.isfile(qfile_cache) and os.path.isfile(afile_cache):

		with open(qfile_cache, 'rb') as f:
			questions = pickle.load(f)
	
		with open(afile_cache, 'rb') as f:
			answers = pickle.load(f)

		print('> Loaded WikiQA cache.')

	else:
		with open(file, encoding="utf8") as data_file:
			source = list(csv.reader(data_file, delimiter="\t", quotechar='"'))
			q_index = 'Q-1'
			ans_sents = {}

			for row in source[1:]:

				if q_index != row[0]:
					answers.append(ans_sents)
					ans_sents = {}
					questions.append(row[1])
					q_index = row[0]

				if len(row[5].split()) <= max_sent_len:	
					ans_sents[row[5]] = row[6]

			answers.append(ans_sents)
			answers = answers[1:]

		with open(qfile_cache, 'wb') as f:
			pickle.dump(questions, f)
		
		with open(afile_cache, 'wb') as f:
			pickle.dump(answers, f)

		print('> No cache found. Pickled WikiQA')

	# for i in range(len(questions)):
	# 	print("Question:", questions[i])
	# 	print("Answers:", answers[i])

	return questions, answers

def qa_vectorize(q, a, glove):
	max_sent_len = 50
	
	q_vector = get_vector_sequence(q, glove, 200)
	a_vector = get_vector_sequence(a, glove, 200)

	q_vector = pad_matrix_with_zeros(q_vector, max_sent_len - len(q.split()))
	a_vector = pad_matrix_with_zeros(a_vector, max_sent_len - len(a.split()))
	# print(q_vector.shape, a_vector.shape)
	# print(" > Vectors Padded")
	return q_vector, a_vector

def get_question_answer_pair_babi(babi):
	from tqdm import tqdm
	from IR import infoRX
	# glove = load_glove(200)

	babi_data = []

	for sample in tqdm(babi, total=len(babi), ncols=75, unit='Sample'):
		line_numbers, context, question, _, support = sample

		tfidf, imp_tokens = infoRX.tf_idf(context, question)

		for i, c in enumerate(context):
			# q_vector, a_vector = qa_vectorize(question, c, glove)

			label = 0
			line_number = line_numbers[i]
			if int(support) == int(line_number):
				label = 1

			word_cnt = 0
			for imp in imp_tokens:
				if imp in c:
					word_cnt += 1

			babi_data.append((question, c, label, tfidf[i], word_cnt))
	return babi_data

def process_babi_for_abcnn(babi):
	from tqdm import tqdm

	samples = []
	
	for line in tqdm(babi, total=len(babi), ncols=75, unit='Line'):
		line_number, data = tuple(line.split(' ', 1))
		data = data.lower()
		if line_number == '1':
			context = []
			line_numbers = []
		
		if '?' in data:
			question, ans_token, support = tuple(data.split(sep='\t'))
			samples.append((line_numbers, context, question, ans_token, support))
		else:
			context.append(data)
			line_numbers.append(line_number)

	return samples

def get_babi_raw_for_abcnn(babi_id, mode):
	babi_raw = []

	babi_map = {
	'1': 'qa1_single-supporting-fact_',
	'2': 'qa2_two-supporting-facts_',
	'3': 'qa3_three-supporting-facts_',
	'4': 'qa4_two-arg-relations_',
	'5': 'qa5_three-arg-relations_',
	'6': 'qa6_yes-no-questions_',
	'7': 'qa7_counting_',
	'8': 'qa8_lists-sets_',
	'9': 'qa9_simple-negation_',
	'10': 'qa10_indefinite-knowledge_',
	'11': 'qa11_basic-coreference_',
	'12': 'qa12_conjunction_',
	'13': 'qa13_compound-coreference_',
	'14': 'qa14_time-reasoning_',
	'15': 'qa15_basic-deduction_',
	'16': 'qa16_basic-induction_',
	'17': 'qa17_positional-reasoning_',
	'18': 'qa18_size-reasoning_',
	'19': 'qa19_path-finding_',
	'20': 'qa20_agents-motivations_'
	}
	babi_name = babi_map[babi_id]
	babi_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/en/{}{}.txt'.format(babi_name, mode))
	
	with open(babi_file, 'r') as f:
		print('  > Getting raw bAbI {} {}'.format(babi_id, mode))
		for line in f:
			babi_raw.append(line.replace('\n', ''))
	
	return babi_raw

def get_babi_for_abcnn(babi_id='1', mode='train'):
	babi = []

	cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/cache/babi_{}_{}.pkl'.format(babi_id, mode))
	if os.path.isfile(cache_file):
		print('> Using cached bAbI {} {} for abcnn'.format(babi_id, mode))
		with open(cache_file, 'rb') as f:
			babi = pickle.load(f)
	
	else:
		print(' > Preparing bAbI {} {} for abcnn'.format(babi_id, mode))
		babi_raw = get_babi_raw_for_abcnn(babi_id, mode)

		print(' > Processing bAbI {} {}'.format(babi_id, mode))
		babi = process_babi_for_abcnn(babi_raw)

		print(' > Getting QA pairs for bAbI {} {}'.format(babi_id, mode))
		babi = get_question_answer_pair_babi(babi)

		print('> Caching the final babi {} {} for abcnn'.format(babi_id, mode))
		with open(cache_file, 'wb') as f:
			pickle.dump(babi, f)

	return babi

def get_question_answer_pair_wikiqa(wikiqa):
	from tqdm import tqdm
	from IR import infoRX

	from nltk.stem.wordnet import WordNetLemmatizer
	lmtzr = WordNetLemmatizer()
	# glove = load_glove(200)

	wikiqa_data = []

	# print('here\n', wikiqa[0])
	for sample in tqdm(wikiqa, total=len(wikiqa), ncols=75, unit='Sample'):
		q, context, label_list = sample
		
		tfidf, imp_tokens = infoRX.tf_idf(context, q)
		
		for i, c in enumerate(context):
			# q_vector, a_vector = qa_vectorize(q, c, glove)

			word_cnt = 0
			for imp in imp_tokens:

				if lmtzr.lemmatize(imp) in [lmtzr.lemmatize(w) for w in c.split()]:
					word_cnt += 1

			wikiqa_data.append((q, c, label_list[i], tfidf[i], word_cnt))
	# print(wikiqa_data[0])
	return wikiqa_data

def process_wikiqa_for_abcnn(wikiqa, mode, max_sent_len=50):
	from tqdm import tqdm
	
	samples = []

	d_id = 'D1'
	if mode == 'test':
		d_id = 'D0'
	context, label_list = [], []
	q_ = ''
	
	for line in tqdm(wikiqa, total=len(wikiqa), ncols=75, unit='Line'):
		_, q, doc_id, _, _, sent, label = tuple(line.split(sep='\t'))
		
		if doc_id != d_id:
			# if len(context) > 0:
			samples.append((q_, context, label_list))
			context = []
			label_list = []
			d_id = doc_id
		
		q_ = q.lower()
		if len(sent.split()) <= max_sent_len:
			context.append(sent.lower())
			label_list.append(int(label))
	
	return samples

def get_wikiqa_raw(mode):
	wikiqa_raw = []

	wikiqa_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/wikiqa/WikiQA-{}.tsv'.format(mode))
	with open(wikiqa_path, 'r') as f:
		print('  > Getting WikiQA {} raw from \'{}\''.format(mode, wikiqa_path))
		for line in f:
			wikiqa_raw.append(line)
		wikiqa_raw = wikiqa_raw[1: ]
	
	return wikiqa_raw

def get_wikiqa_for_abcnn(mode='train'):
	wikiqa = []

	cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/cache/WikiQA_{}.pkl'.format(mode))
	if os.path.isfile(cache_file):
		print(' > Using cached WikiQA {}'.format(mode))
		with open(cache_file, 'rb') as f:
			wikiqa = pickle.load(f)
	
	else:
		print(' > Preparing WikiQA {} for abcnn'.format(mode))
		wikiqa_raw = get_wikiqa_raw(mode)

		# print(wikiqa_raw[0])
		print(' > Processing WikiQA {}'.format(mode))
		wikiqa = process_wikiqa_for_abcnn(wikiqa_raw, mode, 50)
		# print(wikiqa[0])
		print(' > Getting QA pairs for WikiQA {}'.format(mode))
		wikiqa = get_question_answer_pair_wikiqa(wikiqa)
		# print(wikiqa[0])
		print('> Caching WikiQA {}'.format(mode))
		with open(cache_file, 'wb') as f:
			pickle.dump(wikiqa, f)

	return wikiqa

def main():
	url = "https://en.wikipedia.org/wiki/Stanford_University"
	fetch_wiki(url)
	return


if __name__ == '__main__':
	main()
