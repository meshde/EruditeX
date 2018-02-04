import path_utils
import utils


class SICK:
	def get_data():
		PATH = path_utils.get_sick_path()

		with open(PATH,'r') as f:
			lines = [line for line in f]
			lines = lines[1:]

			dataset = []

			for line in lines:
				entry = dict()

				line = line.split('\t')

				entry['A'] = line[1]
				entry['B'] = line[2]
				entry['score'] = float(line[4])

				dataset.append(entry)

		return dataset

	def get_input_tree_single(data, nlp, glove):
		dtree_entry = dict()
		dtne_entry = dict()

		doc1 = nlp(data['A'])
		doc2 = nlp(data['B'])

		dtree_entry['A'] = get_single_sentence_input_dtree(doc1, glove)
		dtree_entry['B'] = get_single_sentence_input_dtree(doc2, glove)
		dtree_entry['score'] = data['score']

		dtne_entry['A'] = get_single_sentence_input_dtne(doc1, glove)
		dtne_entry['B'] = get_single_sentence_input_dtne(doc2, glove)
		dtne_entry['score'] = data['score']

		return dtree_entry, dtne_entry

	def get_input_tree():
		import spacy
		nlp = spacy.load('en')

		sick = SICK.get_data()
		glove = utils.load_glove(200)

		dataset_dtree = []
		dataset_dtne = []

		for data in sick:
			dtree_entry, dtne_entry = SICK.get_input_tree_single(data, nlp, glove)
			dataset_dtree.append(dtree_entry)
			dataset_dtne.append(dtne_entry)
		return dataset_dtree, dataset_dtne

	def get_final_input():
		from os import path as path
		import pickle as pkl
		dataset_dtree, dataset_dtne = [], []
		file_path = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'data/cache/SICK_cache.pkl')
		if path.isfile(file_path):
			with open(file_path, 'rb') as f:
				dataset_dtree, dataset_dtne = pkl.load(f)
			return dataset_dtree, dataset_dtne
		else:
			dataset_dtree, dataset_dtne = SICK.get_input_tree()
			with open(file_path, 'wb') as f:
				pkl.dump((dataset_dtree, dataset_dtne), f)
			return dataset_dtree, dataset_dtne
	

def get_single_sentence_input_dtree(doc, glove):
	dtree_entry = dict()
	sent = utils.get_sentence_from_doc(doc)

	dtree = utils.get_tree_node(sent.root, glove, dim=200)
	word_vectors, parent_indices, is_leaf, dep_tags = dtree.get_rnn_input()

	dtree_entry['word_vectors'] = word_vectors
	dtree_entry['parent_indices'] = parent_indices
	dtree_entry['is_leaf'] = is_leaf
	dtree_entry['dep_tags'] = dep_tags
	dtree_entry['text'] = dtree.get_tree_traversal('text')

	return dtree_entry

def get_single_sentence_input_dtne(doc, glove):
	dtne_entry = dict()

	for ent in doc.ents:
		ent.merge()

	sent = utils.get_sentence_from_doc(doc)

	dtne = utils.get_dtne_node(sent.root, glove, dim=200)
	word_vectors, parent_indices, is_leaf, dep_tags, ent_type  = dtne.get_rnn_input()

	dtne_entry['word_vectors'] = word_vectors
	dtne_entry['parent_indices'] = parent_indices
	dtne_entry['is_leaf'] = is_leaf
	dtne_entry['dep_tags'] = dep_tags
	dtne_entry['ent_type'] = ent_type
	dtne_entry['text'] = dtne.get_tree_traversal('text')

	return dtne_entry 

if __name__ == '__main__':
	SICK.get_final_input()