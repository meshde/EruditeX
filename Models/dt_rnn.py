import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.scan_module import scan

from Helpers import nn_utils


class DT_RNN(object):
	def __init__(self, dep_len=56, dim=50, word_vector_size=200):
		self.dim = dim
		self.word_vector_size = word_vector_size
		self.dep_len = dep_len
		self.set_params()
		# self.theano_build()

	def set_params(self):
		self.W_x =  nn_utils.normal_param(std=0.1, shape=(self.word_vector_size, self.dim))
		self.W_dep = nn_utils.normal_param(std=0.1, shape=(self.dep_len, self.dim, self.dim))
		self.b = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		self.params = [self.W_x, self.W_dep, self.b]
		return

	def get_graph_input(self):
		vectors = T.matrix()
		parent_indices = T.vector()
		is_leaf = T.vector()

		## It is important for dep_tags to be a vector of integers (ivector), or W[dep_tags[idy]] in inner_loop() gives an error "TypeError: Expected an integer"
		dep_tags = T.lvector()

		return [vectors, parent_indices, is_leaf, dep_tags]
	
	def theano_build(self):
		inputs = self.get_graph_input()

		hidden_states, sentence_embedding = self.get_theano_graph(inputs)

		self.get_sentence_embedding = theano.function(inputs, sentence_embedding)
		self.get_hidden_states = theano.function(inputs, hidden_states)

	@staticmethod
	def inner_loop(idy, prev_val, idx, hidden_states, parent_indices, dep_tags, W):
		cond = T.eq(parent_indices[idy], idx)
		temp = ifelse(cond,
					  T.dot(hidden_states[idy], W[dep_tags[idy]]),
					  T.zeros_like(prev_val)
					  )
		val = prev_val + temp
		return val

	@staticmethod
	def outer_loop(idx, hidden_states, vectors, parent_indices, is_leaf,
				   dep_tags, W_x, W_dep, b):

		x,_ = theano.scan(fn=DT_RNN.inner_loop,
						  sequences=T.arange(vectors.shape[0]),
						  outputs_info=T.zeros_like(hidden_states[0]),
						  non_sequences=[idx, hidden_states, parent_indices, dep_tags, W_dep]
						  )
		
		x = x[-1]
		y = T.dot(vectors[idx],W_x) + b
		hidden_state = ifelse(is_leaf[idx],y,x+y)
		res = T.set_subtensor(hidden_states[idx],hidden_state)
		return res

	def get_theano_graph(self, inputs):
		length_of_sentence = inputs[0].shape[0]
		
		hidden_states,_ = theano.scan(fn=self.__class__.outer_loop,
									  sequences=T.arange(length_of_sentence),
									  outputs_info=T.zeros((length_of_sentence,self.dim)),
									  non_sequences=inputs+self.params
									  )

		hidden_states = hidden_states[-1]
		sentence_embedding = hidden_states[-1]

		return hidden_states, sentence_embedding

	def save_params(self,file_name,epochs):
		with open(file_name, 'wb') as save_file:
			pickle.dump(
				obj = {
					'params' : [x.get_value() for x in self.params],
					'epoch' : epochs,
				},
				file = save_file,
				protocol = -1
			)
		return

	def load_params(self,file_name):
		with open(file_name, 'rb') as load_file:
			dict = pickle.load(load_file)
			loaded_params = dict['params']
			for (x, y) in zip(self.params, loaded_params):
				x.set_value(y)
		return


class DTNE_RNN(DT_RNN):
	def __init__(self, dep_len=56, dim=50, word_vector_size=50, ne_len=18):
		self.ne_len	= ne_len
		super().__init__(dep_len, dim, word_vector_size)

	def set_params(self):
		super().set_params()
		self.W_ne = nn_utils.normal_param(std=0.1, shape=(self.dep_len, self.dim))
		self.params.append(self.W_ne)
		return

	def get_graph_input(self):
		inputs = super().get_graph_input()
		ne_indices = T.ivector()
		inputs.append(ne_indices)
		return inputs

	@staticmethod
	def outer_loop(idx, hidden_states, vectors, parent_indices, is_leaf, dep_tags, ne_indices, W_x, W_dep, b, W_ne):
		x,_ = theano.scan(fn=DTNE_RNN.inner_loop, sequences=T.arange(idx), outputs_info=T.zeros_like(hidden_states[0]), non_sequences=[idx,hidden_states,parent_indices,dep_tags,W_dep])
		x = x[-1]
		p = T.dot(vectors[idx],W_x)
		y = ifelse(ne_indices[idx], W_ne[ne_indices[idx]-1], p)
		y = y + b
		hidden_state = ifelse(is_leaf[idx],y,x+y)
		hidden_states = T.set_subtensor(hidden_states[idx],hidden_state)
		return hidden_states




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
