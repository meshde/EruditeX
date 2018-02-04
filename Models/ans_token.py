import theano
import theano.tensor as T
from theano.ifelse import ifelse
import lasagne
from Helpers import nn_utils


class AnsSelect(object):
	def __init__(self,inp_hidden_states_dim,hid_dim=50):

		self.dim=hid_dim
		self.inp_hidden_states_dim=inp_hidden_states_dim

		# Forming the input layer of the answer module
		q_sent_hid=T.vector("Question root node hidden State")
		ans_sent_hid=T.vector("Answer root node hidden state")
		ans_node_hid=T.vector("Answer word node hidden state")
		ans_parent_hid=T.vector("Answer word's parent hidden state")
		ans_sib_hid=T.vector("Answer word's sibling hidden state")
		answer=T.scalar("Answer Probability")



		# Forming the processing layer
		self.W_q=nn_utils.normal_param(std=0.1, shape=(self.dim, self.inp_hidden_states_dim))
		self.W_ans_sent=nn_utils.normal_param(std=0.1, shape=(self.dim, self.inp_hidden_states_dim))
		self.W_ans_node=nn_utils.normal_param(std=0.1, shape=(self.dim, self.inp_hidden_states_dim))
		self.W_ans_parent=nn_utils.normal_param(std=0.1, shape=(self.dim, self.inp_hidden_states_dim))
		self.W_ans_sib=nn_utils.normal_param(std=0.1, shape=(self.dim, self.inp_hidden_states_dim))

		self.b_inp = nn_utils.constant_param(value=0.0, shape=(self.dim))

		self.params=[self.W_q,self.W_ans_sent,self.W_ans_node,self.W_ans_parent,self.W_ans_sib,self.answer]

		# Forming the output layer
		prediction=self.compute(q_sent_hid,ans_sent_hid,ans_node_hid,ans_parent_hid,ans_sib_hid)

		# Forming the updates and loss layer		
		loss=T.sqrt(T.square(prediction) - T.square(answer))
		self.updates=lasagne.updates.adadelta(loss, self.params)

		self.train=theano.function([q_sent_hid,ans_sent_hid,ans_node_hid,ans_parent_hid,ans_sib_hid,answer],[],updates=self.updates)
		self.predict=theano.function([q_sent_hid,ans_sent_hid,ans_node_hid,ans_parent_hid,ans_sib_hid],prediction)

	def compute(self,q_sent_hid,ans_sent_hid,ans_node_hid,ans_parent_hid,ans_sib_hid):
		x=T.dot(self.W_q,q_sent_hid)+T.dot(self.W_ans_sent,ans_sent_hid)+T.dot(self.W_ans_node,ans_node_hid)+T.dot(self.W_ans_parent,ans_parent_hid)+T.dot(self.W_ans_sib,ans_sib_hid)
		out=T.nnet.sigmoid(x+self.b_inp)
		return out

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




		


