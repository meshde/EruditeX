import requests
import numpy as np
import os
import csv
import pickle
import spacy
import trees

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


def get_list_sequence(sentence, glove):
	result = []
	for word in sentence.split():
		result.append(get_vector(word, glove))
	return result


def get_vector_sequence(sentence, glove, dim=50):
	list_sequence = get_list_sequence(sentence, glove)
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

def _process_wikiqa_dataset(file):
	questions = []
	answers = []
	# file = ".\Dataset\WikiQACorpus\WikiQA-dev.tsv"

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

			ans_sents[row[5]] = row[6]

		answers.append(ans_sents)
		answers = answers[1:]

	# for i in range(len(questions)):
	# 	print("Question:", questions[i])
	# 	print("Answers:", answers[i])

	return questions, answers

def main():
	url = "https://en.wikipedia.org/wiki/Stanford_University"
	fetch_wiki(url)
	return


if __name__ == '__main__':
	main()
