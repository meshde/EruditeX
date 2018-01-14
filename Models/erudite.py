import theano
import theano.tensor as T

from Models import dt_rnn
from Helpers import utils

class EruditeX(object):
	def __init__(self, dim=50, word_vector_size=50):
		self.dim = dim
		self.word_vector_size = word_vector_size
		self.dt_rnn = dt_rnn.DT_RNN(dim=dim, word_vector_size=word_vector_size)

	def process_input(self, sentences, question=None):
		vectors_list = []
		parent_indices_list = []
		is_leaf_list = []
		dep_tags_list = []

		dtree_nodes = []

		for sentence in sentences:
			dtree_nodes.append(utils.get_dtree(sentence))

		max_len = max([node.count_nodes() for node in dtree_nodes])

		for node in dtree_nodes:
			vectors,parent_indices,is_leaf,dep_tags = node.get_rnn_input()

			pad_width = max_len - node.count_nodes()
			
			vectors_list.append(utils.pad_matrix_with_zeros(vectors, pad_width=pad_width))
			parent_indices_list.append(utils.pad_vector_with_zeros(parent_indices, pad_width=pad_width))
			is_leaf_list.append(utils.pad_vector_with_zeros(is_leaf, pad_width=pad_width))
			dep_tags_list.append(utils.pad_vector_with_zeros(dep_tags, pad_width=pad_width))

		print(vectors_list)
		print(parent_indices_list)
		print(is_leaf_list)
		print(dep_tags_list)

		return
