import theano
import theano.tensor as T
from theano.ifelse import ifelse

from Models import DT_RNN
from Helpers import utils

class EruditeX(object):
	def __init__(self, dim=50, word_vector_size=50):
		self.dim = dim
		self.word_vector_size = word_vector_size
		self.dt_rnn = DT_RNN(dim=dim, word_vector_size=word_vector_size)

		self.theano_build()


	def theano_build(self):	
		vectors_list = T.tensor3()
		parent_indices_list = T.matrix()
		is_leaf_list = T.matrix()
		dep_tags_list = T.imatrix()

		def dt_rnn_function(idx, vectors_list, parent_indices_list, is_leaf_list, dep_tags_list):
			hidden_states, _ = self.dt_rnn.get_theano_graph(vectors_list[idx], parent_indices_list[idx], is_leaf_list[idx], dep_tags_list[idx])
			return hidden_states

		hidden_states, _ = theano.scan(fn=dt_rnn_function, sequences=T.arange(vectors_list.shape[0]), non_sequences=[vectors_list,parent_indices_list,is_leaf_list,dep_tags_list])

		def inner_loop(hidden_state):
			return hidden_state, theano.scan_module.until(T.all(T.eq(hidden_state, T.zeros_like(hidden_state))))

		def get_sentence_embeddings_function(hidden_states):
			sentence_embedding, _ = theano.scan(fn=inner_loop,sequences=hidden_states)

			sentence_embedding = ifelse(T.all(T.eq(sentence_embedding[-1], T.zeros_like(sentence_embedding[-1]))), sentence_embedding[-2], sentence_embedding[-1])

			return sentence_embedding

		sentence_embeddings, _ = theano.scan(fn=get_sentence_embeddings_function, sequences=hidden_states)

		self.get_hidden_states = theano.function([vectors_list, parent_indices_list, is_leaf_list, dep_tags_list], hidden_states)

		self.get_sentence_embeddings = theano.function([vectors_list, parent_indices_list, is_leaf_list, dep_tags_list], sentence_embeddings)

		self.get_sentence_embeddings_with_hidden_states = theano.function([vectors_list, parent_indices_list, is_leaf_list, dep_tags_list], [sentence_embeddings, hidden_states])
		
		return


	def process_context(self, sentences):
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

		self.vectors_list = vectors_list
		self.parent_indices_list = parent_indices_list
		self.is_leaf_list = is_leaf_list
		self.dep_tags_list = dep_tags_list

		return