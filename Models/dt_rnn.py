import theano
import theano.tensor as T
from theano.ifelse import ifelse

from Helpers import nn_utils


class DT_RNN(object):
	def __init__(self, dep_len=56, dim=50, word_vector_size=50):
		self.dim = dim
		self.word_vector_size = word_vector_size
		self.dep_len = dep_len

		self.W_x =  nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
		self.W_dep = nn_utils.normal_param(std=0.1, shape=(self.dep_len, self.dim, self.dim))
		self.b = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		self.theano_build()

	def theano_build(self):
		vectors = T.matrix()
		parent_indices = T.vector()
		is_leaf = T.vector()

		## It is important for dep_tags to be a vector of integers (ivector), or W[dep_tags[idy]] in inner_loop() gives an error "TypeError: Expected an integer"
		dep_tags = T.ivector()

		# hidden_states = T.zeros((vectors.shape[0],self.dim))

		hidden_states, sentence_embedding = self.get_theano_graph(vectors, parent_indices, is_leaf, dep_tags)

		self.get_sentence_embedding = theano.function([vectors,parent_indices,is_leaf,dep_tags],sentence_embedding)
		self.get_hidden_states = theano.function([vectors,parent_indices,is_leaf,dep_tags],hidden_states)

		return

	def get_theano_graph(self, vectors, parent_indices, is_leaf, dep_tags):

		def inner_loop(idy, prev_val, idx, hidden_states, parent_indices, dep_tags, W):
			temp = ifelse(T.eq(parent_indices[idy],idx),T.dot(hidden_states[idy],W[dep_tags[idy]]),T.zeros_like(prev_val))
			val = prev_val + temp
			return val

		def outer_loop(idx, hidden_states, vectors, parent_indices, is_leaf, dep_tags, W_x, W_dep, b):
			x,_ = theano.scan(fn=inner_loop, sequences=T.arange(idx), outputs_info=T.zeros_like(hidden_states[0]), non_sequences=[idx,hidden_states,parent_indices,dep_tags,W_dep])
			x = x[-1]
			y = T.dot(vectors[idx],W_x) + b
			hidden_state = ifelse(is_leaf[idx],y,x+y)
			hidden_states = T.set_subtensor(hidden_states[idx],hidden_state)
			return hidden_states

		hidden_states,_ = theano.scan(fn=outer_loop, sequences=T.arange(vectors.shape[0]), outputs_info=T.zeros((vectors.shape[0],self.dim)), non_sequences=[vectors,parent_indices,is_leaf,dep_tags,self.W_x,self.W_dep,self.b])
		hidden_states = hidden_states[-1]
		sentence_embedding = hidden_states[-1]

		return hidden_states, sentence_embedding


import numpy as np

def np_dt_rnn(vectors, parent_indices, is_leaf, dep_tags, W_x, W_dep, b):
	hidden_states = []
	for i in range(len(parent_indices)):
		total = np.dot(vectors[i], W_x) + b
		if not is_leaf[i]:
			for j in range(len(parent_indices)):
				if parent_indices[j] == i and i != j:
					total += np.dot(hidden_states[j], W_dep[dep_tags[j]])
		hidden_states.append(total)
	return hidden_states







