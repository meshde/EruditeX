import theano
import theano.tensor as T
from Helpers import nn_utils


class DT_RNN(object):
	def __init__(self,dim=50,word_vector_size=50):
		self.dim = dim
		self.word_vector_size = word_vector_size
		self.dep_len = 20 # Mit please handle dep tags

		self.W_x =  nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
		self.W_dep = nn_utils.normal_param(std=0.1, shape=(self.dep_len, self.dim, self.dim))
		self.b = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		self.theano_build()

	def theano_build(self):
		vectors = T.matrix()
		parent_indices = T.vector()
		is_leaf = T.vector()
		dep_tags = T.vector()

		hidden_states = T.zeros((vectors.shape[0],self.dim))

		def inner_loop(idy,prev_val,idx,hidden_states,parent_indices,dep_tags,W):
			temp = ifelse(T.eq(parent_indices[idy],idx),T.dot(W[dep_tags[idy]],hidden_states[idy]),T.zeros_like(prev_val))
			val = prev_val + temp
			return val

		def outer_loop(idx,hidden_states,vectors,parent_indices,is_leaf,dep_tags,W_x,W_dep,b):
			x,_ = thano.scan(fn=inner_loop,sequences=T.arange(vectors.shape[0]),non_sequences=[hidden_states,parent_indices,dep_tags,W_dep])
			x = x[-1]
			y = T.dot(W_x,vectors[idx]) + b
			hidden_state = ifesle(is_leaf[idx],y,x+y)
			hidden_states = T.set_subtensor(hidden_states[idx],hidden_state)
			return hidden_states

		hidden_states,_ = theano.scan(fn=outer_loop,sequences=T.arange(vectors.shape[0]),non_sequences=[vectors,parent_indices,is_leaf,dep_tags,self.W_x,self.W_dep,self.b])
		hidden_states = hidden_states[-1]
		sentence_embedding = hidden_states[-1]

		self.get_sentence_embedding = theano.function([vectors,parent_indices,is_leaf,dep_tags],sentence_embedding)

		return