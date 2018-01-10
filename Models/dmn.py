
import random
import numpy as np

import sys

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from theano import pp

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import pickle as pickle

from Helpers import utils
from Helpers import nn_utils

from Models import SentEmbd

floatX = theano.config.floatX

class DMN:
	def __init__(self, babi_train_raw, babi_test_raw, word2vec,word_vector_size,dim, mode, answer_module, input_mask_mode, memory_hops, l2,normalize_attention, answer_vec, debug,sentEmbdLoadState,sentEmbdType="basic",**kwargs):
		self.vocab = {}
		self.ivocab = {}
		self.debug = debug

		self.word2vec = word2vec
		self.word_vector_size = word_vector_size
		self.dim = dim
		self.mode = mode
		self.answer_module = answer_module
		self.input_mask_mode = input_mask_mode
		self.memory_hops = memory_hops
		self.l2 = l2
		self.normalize_attention = normalize_attention
		self.answer_vec = answer_vec
		self.sentEmbdType=sentEmbdType
		if(self.mode!='deploy'):
			self.train_input, self.train_q, self.train_answer, self.train_input_mask = self._process_input(babi_train_raw)
			self.test_input, self.test_q, self.test_answer, self.test_input_mask = self._process_input(babi_test_raw)
			self.vocab_size = len(self.vocab)
			print(self.vocab_size)
		elif self.mode =='deploy':
			self.train_input, self.train_q, self.train_answer, self.train_input_mask = self._process_input(babi_train_raw)
			self.vocab_size = len(self.vocab)
			print(self.vocab_size)
			# print(self.train_input.shape)
			# print(self.train_q.shape)
			# print(self.train_input_mask.shape)

		#Setting up pre-trained Sentence Embedder for question and input module:
		if self.mode != 'deploy': print("==> Setting up pre-trained Sentence Embedder")       
		if self.sentEmbdType=="basic":
			self.sent_embd=SentEmbd.SentEmbd_basic(self.word_vector_size,self.dim)
		else:
			dep_tags=utils.load_dep_tags
			self.sent_embd=SentEmbd.SentEmbd_syntactic(50,hid_dim,len(dep_tags)) #TODO: Dependency Tags
		self.sent_embd.load_params(sentEmbdLoadState)

		self.input_var = T.matrix('input_var')
		self.q_var = T.vector('question_var')
		if self.answer_vec == 'word2vec':
			self.answer_var = T.vector('answer_var')
		else:
			self.answer_var = T.iscalar('answer_var')
		self.input_mask_var = T.ivector('input_mask_var')

		if self.answer_vec == 'one_hot' or self.answer_vec == 'index':
			self.answer_size = self.vocab_size
		elif self.answer_vec == 'word2vec':
			self.answer_size = self.word_vector_size
		else:
			raise Exception("Invalid answer_vec type")


		#Setting up Untrained Memory module
		if self.mode != 'deploy': print("==> Creating parameters for memory module")
		self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 2))
		self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
		self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
		self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))


		if self.mode != 'deploy': print("==> Building episodic memory module (fixed number of steps: %d)" % self.memory_hops)
		memory = [self.q_var.copy()]
		for iter in range(1, self.memory_hops + 1):
			current_episode = self.new_episode(memory[iter - 1])
			memory.append(self.GRU_update(memory[iter - 1], current_episode,
										  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
										  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
										  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))

		last_mem = memory[-1]

		if self.mode != 'deploy': print("==> Building answer module")

		self.W_a = nn_utils.normal_param(std=0.1, shape=(self.answer_size, self.dim))

		if self.answer_module == 'feedforward':
			self.prediction = nn_utils.softmax(T.dot(self.W_a, last_mem))
		# elif self.answer_module == 'recurrent':
		#     self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.answer_size))
		#     self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		#     self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		#     self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.answer_size))
		#     self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		#     self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		#     self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.answer_size))
		#     self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
		#     self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

		#     def answer_step(prev_a, prev_y):
		#         a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
		#                           self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res,
		#                           self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
		#                           self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
		#         y = T.dot(self.W_a, a)
		#         if self.answer_vec == 'one_hot' or self.answer_vec == 'index':
		#             y = nn_utils.softmax(y)
		#         return [a, y]

		#     # TODO: add conditional ending
		#     dummy = theano.shared(np.zeros((self.answer_size, ), dtype=floatX))
		#     results, updates = theano.scan(fn=answer_step,
		#         outputs_info=[last_mem, T.zeros_like(dummy)],
		#         n_steps=1)
		#     self.prediction = results[1][-1]

		else:
			raise Exception("invalid answer_module")


		if self.mode != 'deploy': print("==> Collecting all parameters to be trained")
		self.params = [self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
				  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
				  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid,
				  self.W_b, self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]

		# if self.answer_module == 'recurrent':
		#     self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res,
		#                       self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
		#                       self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]

		if self.mode != 'deploy': print("==> Building loss layer and computing updates")
		if debug:
			print('Prediction dim:',self.prediction.dimshuffle('x', 0).ndim)
			print('Answer dim:',self.answer_var.ndim)
		if self.answer_vec == 'word2vec':
			self.loss_ce = nn_utils.cosine_proximity_loss(self.prediction.dimshuffle('x', 0), T.stack([self.answer_var]))[0][0]
		else:
			self.loss_ce = T.nnet.categorical_crossentropy(self.prediction.dimshuffle('x', 0), T.stack([self.answer_var]))[0]
		if self.l2 > 0:
			self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
		else:
			self.loss_l2 = 0

		self.loss = self.loss_ce + self.loss_l2

		if debug: print(self.loss.ndim)
		# if self.debug: print(self.loss.eval({self.input_var:self.train_input,self.q_var:self.train_q,self.answer_var:self.train_answer,self.input_mask_var:self.train_input_mask}))
		updates = lasagne.updates.adadelta(self.loss, self.params)

		if self.mode == 'deploy':
			self.deploy_fn = theano.function(inputs=[self.input_var, self.q_var],outputs=[self.prediction])

		else:
			if self.mode == 'train':
				print("==> compiling train_fn")
				self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var],
										   outputs=[self.prediction, self.loss],
										   updates=updates)

			print("==> compiling test_fn")
			self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var],
									  outputs=[self.prediction, self.loss, self.input_var, self.q_var, last_mem])


			if self.mode == 'train':
				print("==> computing gradients (for debugging)")
				gradient = T.grad(self.loss, self.params)
				self.get_gradient_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var], outputs=gradient)

	def get_SentenceVecs(self,sentences):
		# print(np.array(sentences).shape)
		if self.sentEmbdType == 'basic':
			sentVecs=self.sent_embd.predict(np.array(sentences))
		elif self.sentEmbdType == 'advanced':
			print("TODO")
			#TODO
		return sentVecs




	def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,W_upd_in, W_upd_hid, b_upd,W_hid_in, W_hid_hid, b_hid):
		z = T.nnet.sigmoid(T.dot(W_upd_in, x) + T.dot(W_upd_hid, h) + b_upd)
		r = T.nnet.sigmoid(T.dot(W_res_in, x) + T.dot(W_res_hid, h) + b_res)
		_h = T.tanh(T.dot(W_hid_in, x) + r * T.dot(W_hid_hid, h) + b_hid)
		return z * h + (1 - z) * _h

	def new_attention_step(self, ct, prev_g, mem, q_q):
		cWq = T.stack([T.dot(T.dot(ct, self.W_b), q_q)])
		cWm = T.stack([T.dot(T.dot(ct, self.W_b), mem)])
		z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, T.abs_(ct - q_q), T.abs_(ct - mem), cWq, cWm])

		l_1 = T.dot(self.W_1, z) + self.b_1
		l_1 = T.tanh(l_1)
		l_2 = T.dot(self.W_2, l_1) + self.b_2
		G = T.nnet.sigmoid(l_2)[0]
		return G


	def new_episode_step(self, ct, g, prev_h):
		gru = self.GRU_update(prev_h, ct,
							 self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
							 self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
							 self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid)

		h = g * gru + (1 - g) * prev_h
		return h


	def new_episode(self, mem):
		g, g_updates = theano.scan(fn=self.new_attention_step,
			sequences=self.input_var,
			non_sequences=[mem, self.q_var],
			outputs_info=T.zeros_like(self.input_var[0][0]))

		if (self.normalize_attention):
			g = nn_utils.softmax(g)

		e, e_updates = theano.scan(fn=self.new_episode_step,
			sequences=[self.input_var, g],
			outputs_info=T.zeros_like(self.input_var[0]))

		return e[-1]

	def save_params(self, file_name, epoch, **kwargs):
		with open(file_name, 'wb') as save_file:
			pickle.dump(
				obj = {
					'params' : [x.get_value() for x in self.params],
					'epoch' : epoch,
					'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
				},
				file = save_file,
				protocol = -1
			)


	def load_state(self, file_name):
		if self.mode != 'deploy': print("==> loading state %s" % file_name)
		with open(file_name, 'rb') as load_file:
			dict = pickle.load(load_file)
			loaded_params = dict['params']
			for (x, y) in zip(self.params, loaded_params):
				x.set_value(y)



	def _process_input(self, data_raw):
		questions = []
		inputs = []
		answers = []
		input_masks = []
		for x in data_raw:
			inp = x["C"].lower().split(' ')
			inp = [w for w in inp if len(w) > 0]
			q = x["Q"].lower().split(' ')
			q = [w for w in q if len(w) > 0]
			print(q)
			# print("Contenxt:",inp)
			# print("Question:",q)
			# print("Answer:",x["A"])


			inp_vector = [utils.process_word(word = w,
										word2vec = self.word2vec,
										vocab = self.vocab,
										ivocab = self.ivocab,
										word_vector_size = self.word_vector_size,
										to_return = "word2vec") for w in inp]


			# print(inp)
			# print([np.array(wvec).shape for wvec in inp_vector])
			# print(np.array(inp_vector).shape)

			q_vector = [utils.process_word(word = w,
										word2vec = self.word2vec,
										vocab = self.vocab,
										ivocab = self.ivocab,
										word_vector_size = self.word_vector_size,
										to_return = "word2vec") for w in q]

			
			inputs.append(np.vstack(inp_vector).astype(floatX))

			questions.append(np.vstack(q_vector).astype(floatX))
			if self.mode != 'deploy':
				answers.append(utils.process_word(word = x["A"],
												word2vec = self.word2vec,
												vocab = self.vocab,
												ivocab = self.ivocab,
												word_vector_size = self.word_vector_size,
												to_return = self.answer_vec))
			# NOTE: here we assume the answer is one word!
			if self.input_mask_mode == 'word':
				input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32))
			elif self.input_mask_mode == 'sentence':
				input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
			else:
				raise Exception("invalid input_mask_mode")

		return inputs, questions, answers, input_masks

	def get_batches_per_epoch(self, mode):
		if mode == 'train':
			return len(self.train_input)
		elif mode == 'test':
			return len(self.test_input)
		else:
			return 1


	def shuffle_train_set(self):
		print("==> Shuffling the train set")
		combined = list(zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask))
		random.shuffle(combined)
		self.train_input, self.train_q, self.train_answer, self.train_input_mask = zip(*combined)



	def step(self, batch_index, mode):
		if mode == "train" and self.mode == "test":
			raise Exception("Cannot train during test mode")

		if mode == "train":
			theano_fn = self.train_fn
			inputs = self.train_input
			qs = self.train_q
			answers = self.train_answer
			input_masks = self.train_input_mask
		elif mode == "test":
			theano_fn = self.test_fn
			inputs = self.test_input
			qs = self.test_q
			answers = self.test_answer
			input_masks = self.test_input_mask
		else:
			raise Exception("Invalid mode")

		inp = inputs[batch_index]
		q = qs[batch_index]
		ans = answers[batch_index]
		input_mask = input_masks[batch_index]

		skipped = 0
		grad_norm = float('NaN')
		inp_vec=self.get_SentenceVecs(inp)
		inp_vec=inp_vec.take(input_mask, axis=0)
		q_vec=self.get_SentenceVecs(q)
		q_vec=q_vec[-1]

		if mode == 'train':
			gradient_value = self.get_gradient_fn(inp_vec, q_vec, ans)
			grad_norm = np.max([utils.get_norm(x) for x in gradient_value])

			if (np.isnan(grad_norm)):
				print("==> gradient is nan at index %d." % batch_index)
				print("==> skipping")
				skipped = 1

		if skipped == 0:
			ret = theano_fn(inp_vec, q_vec, ans)
		else:
			ret = [-1, -1]

		param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])

		return {"prediction": np.array([ret[0]]),
				"answers": np.array([ans]),
				"current_loss": ret[1],
				"skipped": skipped,
				"log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
				}


	def step_deploy(self):
		inputs=self.train_input
		q=self.train_q
		input_mask=self.train_input_mask
		inp_vec=self.get_SentenceVecs(inputs[0])
		inp_vec=inp_vec.take(input_mask, axis=0)
		inp_vec=inp_vec[0]
		q_vec=self.get_SentenceVecs(q[0])
		q_vec=q_vec[-1]
		print(inp_vec.shape)
		print(q_vec.shape)
		prediction = self.deploy_fn(inp_vec,q_vec)
		ans_ind=np.argmax(prediction)
		print(ans_ind)
		print(self.ivocab)
		print(self.ivocab[ans_ind])
		return prediction




