from Helpers import utils

def test_dt_rnn():
	import numpy as np
	from Models import DT_RNN
	from Models import np_dt_rnn

	model = DT_RNN(dim=3, word_vector_size=3)

	np_W_dep = model.W_dep.get_value()
	np_W_x = model.W_x.get_value()
	np_b = model.b.get_value()

	sentence = "welcome to my house"
	dtree = utils.get_dtree(sentence, dim=3)
	vectors,parent_indices,is_leaf,dep_tags = dtree.get_rnn_input()

	np_ans = np_dt_rnn(vectors, parent_indices, is_leaf, dep_tags, np_W_x, np_W_dep, np_b)
	theano_ans = model.get_hidden_states(vectors, parent_indices, is_leaf, dep_tags)

	print(np_ans)
	print(theano_ans)

	assert(np.allclose(np_ans, theano_ans, rtol=1e-04, atol=1e-07))

	return
