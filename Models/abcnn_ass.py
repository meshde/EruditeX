'''
Attention-Based Bi CNN for Answer Sentence Selection from context.

'''

import tensorflow as tf
import numpy as np
import pickle
import sys
import time
import datetime
from tqdm import tqdm

sys.path.append('../')
from os import path as path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To remove the tensorflow warnings completely

from Helpers import utils
from IR import infoRX

# from nltk.corpus import stopwords
# from nltk import word_tokenize
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.spatial.distance import cdist


class abcnn_model:

	def __init__(self):

		# Hyperparameters

		self.vector_dim = 200  # vector_dim
		self.max_sent_len = 50  # max_sent_length
		self.batch_size = 100
		self.filter_size = 4  # filter_size
		self.n_filters = 50  # num_filters
		self.learning_rate = 0.05  # learning_rate
		self.pad_len = self.filter_size - 1
		self.save_state = 10 # will save state every 10 questions
		self.predict_label = None

		self.q = tf.placeholder(tf.float32, [self.max_sent_len, self.vector_dim], 'question')
		self.a = tf.placeholder(tf.float32, [self.max_sent_len, self.vector_dim], 'answer')
		self.label = tf.placeholder(tf.float32, name='label')
		self.word_cnt = tf.placeholder(tf.float32, name='word_cnt')
		self.tfidf = tf.placeholder(tf.float32, name='tfidf')

		# self.Q_ = tf.placeholder(tf.float32, [self.vector_dim])
		# self.A_ = tf.placeholder(tf.float32, [self.vector_dim])

		# [self.max_sent_len, self.vector_dim] [self.vector_dim, self.max_sent_len]
		self.W_q = tf.Variable(tf.random_normal([self.max_sent_len, self.vector_dim]))
		self.W_a = tf.Variable(tf.random_normal([self.max_sent_len, self.vector_dim]))

		self.W_lr = tf.Variable(tf.random_normal([3]))
		self.B_lr = tf.Variable(tf.random_normal([1]))

	def get_pool_size(self):
		return self.max_sent_len


	def pairwise_euclidean_dist(self, m0, m1):

		ted = tf.TensorArray(dtype=tf.float32, size=self.max_sent_len)
		ted.unstack(tf.zeros([self.max_sent_len, self.max_sent_len]))

		c1 = lambda i, j, m0, m1, ta: j < self.max_sent_len

		def body1(i, j, m0, m1, ta):
			x = tf.reciprocal(1 + tf.sqrt(tf.reduce_sum(tf.square(m0[i] - m1[j]))))
			to = ta.write(j, x)
			return i, j + 1, m0, m1, to

		c0 = lambda i, m0, m1, ted: i < self.max_sent_len

		def body0(i, m0, m1, ted):
			ta = tf.TensorArray(dtype=tf.float32, size=self.max_sent_len)
			ta.unstack(tf.zeros([self.max_sent_len]))

			tb = tf.while_loop(c1, body1, loop_vars=[i, 0, m0, m1, ta])[-1]
			tc = tb.stack()
			ted = ted.write(i, tc)
			return i + 1, m0, m1, ted

		ta_final = tf.while_loop(c0, body0, loop_vars=[0, m0, m1, ted])[-1]
		result = ta_final.stack()
		# print("Attn:", result.shape)

		return result


	def attention_pooling(self, feature_map, attn_, axis):

		attn = tf.reduce_sum(attn_, axis=axis)
		# print(feature_map.get_shape(), attn.get_shape())

		ta = tf.TensorArray(dtype=tf.float32, size=self.get_pool_size())

		def body(i, w, a, attn, ta):
			def c1(j, jw, a, attn, sumi):
				return j < jw

			def b1(j, jw, a, attn, sumi):
				v = attn[i]
				q = a[j]
				return j + 1, jw, a, attn, sumi + (q * v)
				# return i+1, w, a, attn, sumi + (a[i] * attn[i])

			res = tf.while_loop(c1, b1, loop_vars=[i, i + w, a, attn, tf.zeros_like(a[0])])[-1]

			return i + 1, w, a, attn, ta.write(i, res)

		def cond(i, w, a, attn, ta):
			return i < self.max_sent_len

		result = tf.while_loop(cond, body, loop_vars=[0, self.filter_size, feature_map, attn, ta])[-1]
		return result.stack()


	def model(self):

		attn_ = self.pairwise_euclidean_dist(self.q, self.a)
		print(" > Attention matrix:", attn_.get_shape())

		# attn_ = tf.cast(euclidean_distances(q_vector, a_vector), tf.float32)
		# attn_ = tf.cast( tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(q_vector, a_vector)), 1)), tf.float32)
		# attn_ = cdist(q_vector, a_vector)

		# print(q_vector.get_shape(), a_vector.get_shape())
		# print(type(q_vector), type(a_vector))
		# print(attn_.get_shape(), self.W_q.get_shape(), self.W_a.get_shape())

		q_feature_ = tf.matmul(attn_, self.W_q)
		a_feature_ = tf.matmul(attn_, self.W_a, transpose_a=True)

		q_vector = tf.stack([q_feature_, self.q], 2)
		a_vector = tf.stack([a_feature_, self.a], 2)

		print(" > Stack:", q_vector.get_shape(), a_vector.get_shape())

		# Expanding dimensions for conv input
		q_vector = tf.reshape(q_vector, [1, self.max_sent_len, self.vector_dim, 2])
		a_vector = tf.reshape(a_vector, [1, self.max_sent_len, self.vector_dim, 2])
		# print(q_vector.get_shape(), a_vector.get_shape())

		# Conv parameter definition
		# filter_4d = tf.constant(np.ones((self.filter_size, self.vector_dim, 2, 1)), dtype=tf.float32)
		# kernel_4d = tf.reshape(filter_4d, [filter_height, filter_width, in_channels, out_channels])
		# strides_2d = [1, 1, 1, 1]


		# Convolutional Layer1
		q_conv = tf.layers.conv2d(
			inputs=q_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, 1],
			padding="valid",
			activation=tf.nn.relu)

		a_conv = tf.layers.conv2d(
			inputs=a_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, 1],
			padding="valid",
			activation=tf.nn.relu)

		print(" > Conv1 Output:", a_conv.get_shape(), q_conv.get_shape())

		# Padding before pooling
		padding = tf.constant([[self.pad_len, self.pad_len], [0, 0], [0, 0]])
		# assert padding.get_shape() == (3,2)
		q_vector = tf.pad(tf.squeeze(q_conv), padding)
		a_vector = tf.pad(tf.squeeze(a_conv), padding)

		print(" > Pad1 Output:", q_vector.get_shape(), a_vector.get_shape())

		attn_ = self.pairwise_euclidean_dist(q_vector, a_vector)
		print(" > Attention matrix:", attn_.get_shape())


		# Attention-based Pooling
		# q_vector = self.attention_pooling(q_vector, attn_)
		# a_vector = self.attention_pooling(a_vector, tf.transpose(attn_))
		q_vector = self.attention_pooling(q_vector, attn_, 0)
		a_vector = self.attention_pooling(a_vector, attn_, 1)

		print(" > Attn_Pool Output:", q_vector.get_shape(), a_vector.get_shape())

		q_feature_ = tf.matmul(attn_, self.W_q)
		a_feature_ = tf.matmul(attn_, self.W_a, transpose_a=True)

		# q_vector = tf.stack([q_feature_, q_vector], 2)
		# a_vector = tf.stack([a_feature_, a_vector], 2)

		q_feature_ = tf.reshape(q_feature_, [self.max_sent_len, self.vector_dim, 1])
		a_feature_ = tf.reshape(a_feature_, [self.max_sent_len, self.vector_dim, 1])

		q_vector = tf.concat([q_feature_, q_vector], 2)
		a_vector = tf.concat([a_feature_, a_vector], 2)

		# Expanding dimensions for conv input
		q_vector = tf.reshape(q_vector, [1, self.max_sent_len, self.vector_dim, 51])
		a_vector = tf.reshape(a_vector, [1, self.max_sent_len, self.vector_dim, 51])

		# Convolutional Layer2

		# q_conv = tf.nn.conv2d(
		# 	input=q_vector,
		# 	filter=filter_4d,
		# 	strides=strides_2d,
		# 	padding="SAME")
		# a_conv = tf.nn.conv2d(
		# 	input=a_vector,
		# 	filter=filter_4d,
		# 	strides=strides_2d,
		# 	padding="SAME")

		q_conv = tf.layers.conv2d(
			inputs=q_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, 1],
			padding="valid",
			activation=tf.nn.relu)

		a_conv = tf.layers.conv2d(
			inputs=a_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, 1],
			padding="valid",
			activation=tf.nn.relu)

		# Padding before pooling
		# q_vector = tf.pad(tf.reduce_sum(tf.squeeze(q_conv), 2), padding)
		# a_vector = tf.pad(tf.reduce_sum(tf.squeeze(a_conv), 2), padding)

		q_vector = tf.pad(tf.squeeze(q_conv), padding)
		a_vector = tf.pad(tf.squeeze(a_conv), padding)

		print(" > Conv2 Output:", q_vector.get_shape(), a_vector.get_shape())

		attn_ = self.pairwise_euclidean_dist(q_vector, a_vector)

		print(" > Attention:", attn_.get_shape())

		q_vector = self.attention_pooling(q_vector, attn_, 0)
		a_vector = self.attention_pooling(a_vector, attn_, 1)

		print(" > Attn_Pool Output:", q_vector.get_shape(), a_vector.get_shape())

		q_vector = tf.reshape(q_vector, [1, self.max_sent_len, self.vector_dim, 50])
		a_vector = tf.reshape(a_vector, [1, self.max_sent_len, self.vector_dim, 50])

		# print(q_vector.get_shape(), a_vector.get_shape())

		pool1 = tf.layers.average_pooling2d(inputs=q_vector, pool_size=[self.max_sent_len, self.vector_dim],
		                                    strides=1)
		pool2 = tf.layers.average_pooling2d(inputs=a_vector, pool_size=[self.max_sent_len, self.vector_dim],
		                                    strides=1)

		print(" > Model Output:", pool1.get_shape(), pool2.get_shape())

		optimizer = tf.train.AdamOptimizer(self.learning_rate)

		cos_sim = tf.losses.cosine_distance(pool1, pool2, 0)
		# print(" > Cosine:", cos_sim.dtype)

		features = tf.stack([cos_sim, self.word_cnt, self.tfidf])

		output_layer = tf.add(tf.tensordot(features, self.W_lr, 1), self.B_lr)

		output_layer_test = tf.sigmoid(output_layer)
		# output_layer += 1e-7

		# cross entropy loss
		loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=output_layer)
		# loss = -((self.label * tf.log(output_layer)) + ((1 - self.label) *
                                                 # tf.log(1 - output_layer)))

		# correct = tf.equal(self.label, output_layer)
		# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		train_step = optimizer.minimize(loss)

		return train_step, output_layer, loss, output_layer_test



	def get_score(self, mode):

		mark_init = time.time()
		score = 0
		p_score = 0
		p_instances = 0
		instances = 0
		pred_labl = -1
		one = tf.constant(1, dtype=tf.float32)
		train_step, output_layer, loss, output_layer_test = self.model()
		glove = utils.load_glove(200)
		saver = tf.train.Saver()
		
		filename = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'states/abcnn/state_')

		q_list, a_list = utils._process_wikiqa_dataset(mode, self.max_sent_len)

		print(" > Dataset initialized. | Elapsed:", time.time() - mark_init)
		# q_list = q_list[:20]
		# a_list = a_list[:20]

		with tf.Session() as sessn:
			sessn.run(tf.global_variables_initializer())

			if mode == 'test':
				with open(filename + 'r.txt', 'rb') as fp:
					now, _ = pickle.load(fp)
					file_path = filename + now + '.ckpt'
					saver.restore(sessn, path.join(path.dirname(path.dirname(path.realpath(__file__))), file_path)) 
					print('> Model restored from @ ', now)

			try:
				if mode == 'train':
					with open(filename + 'r.txt', 'rb') as fp:
						name, q_number = pickle.load(fp)
						if name == 'temp':
							saver.restore(sessn, path.join(path.dirname(path.dirname(path.realpath(__file__))), 'states/abcnn/state_temp.ckpt'))
							q_list = q_list[q_number:]
							a_list = a_list[q_number:]
			except:
				pass

			for i in tqdm(range(len(q_list)), total=len(q_list), ncols=75, unit='Question'):
				q = q_list[i]
				tfidf, imp_tokens = infoRX.tf_idf([str(a) for a in a_list[i].keys()], q)
				j = 0

				for a in tqdm(a_list[i].keys(), total=len(a_list[i].keys()), ncols=75, unit='Answer'):

					mark_start = time.time()
					word_cnt = 0
					instances += 1
					label_ = a_list[i][a] # 0 if not answer and 1 if is answer

					for x in imp_tokens:
						if x in a:
							word_cnt += 1

					q_vector = utils.get_vector_sequence(q, glove, 200)
					a_vector = utils.get_vector_sequence(a, glove, 200)
					q_vector = utils.pad_matrix_with_zeros(q_vector,
					                                       self.max_sent_len - len(q.split()))
					a_vector = utils.pad_matrix_with_zeros(a_vector,
					                                       self.max_sent_len - len(a.split()))
					# print(q_vector.shape, a_vector.shape)
					# print(" > Vectors Padded")

					input_dict = {self.q: q_vector, self.a: a_vector, self.label: label_,
					              self.word_cnt: word_cnt, self.tfidf: tfidf[j]}

					if mode == 'train':
						_, output, l, self.predict_label = sessn.run([train_step, output_layer, loss, output_layer_test], feed_dict=input_dict)
					
					# else:
					# 	l = sessn.run(output_layer_test, feed_dict=input_dict)

					if self.predict_label[0] > 0.5:
						pred_labl = 1
					else:
						pred_labl = 0

					if label_ == 1:
						p_instances += 1

					# print(score, self.predict_label, label_)
					if pred_labl == int(label_):
						# print(score, label_)
						score += 1
						if label_ == 1:
							p_score += 1

					with open('result_abcnn.txt', 'a') as f:
						itr_res = str('> QA_Iteration' + str(i) + '-' + str(j) + '| Output Layer: ' + str(self.predict_label) + ' | Loss:' + str(l) + ' | Label: ' + str(label_) + '| Elapsed: {0:.2f}'.format(time.time() - mark_start) + '\n')
						f.write(itr_res)
					j += 1

				if i % self.save_state == 0:
					if mode == 'train':
						file_path = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'states/abcnn/state_temp.ckpt')
						saver.save(sessn, path.join(path.dirname(path.dirname(path.realpath(__file__))), file_path))
						with open(filename + 'r.txt', 'wb') as fp:
							pickle.dump(('temp', i), fp)

					accuracy = (score / instances) * 100 
					p_accuracy = (p_score / p_instances) * 100 

					score, instances = 0, 0
					p_score, p_instances = 0, 0

					with open('result_abcnn.txt', 'a') as f:
						itr_res = str('> Accuracy: {0:.2f}'.format(accuracy) + '\n')
						itr_res += str('> True_P accuracy: {0:.2f}'.format(p_accuracy) + '\n')
						f.write(itr_res)


			if mode == 'train':
				now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
				file_path = filename + now + '.ckpt'
				saver.save(sessn, path.join(path.dirname(path.dirname(path.realpath(__file__))), file_path)) 
				print('\n> Model state saved @ ', now)
				with open(filename + 'r.txt', 'wb') as fp:
					pickle.dump(now, fp)

# model verification
if __name__ == '__main__':
	selector = abcnn_model()
	selector.get_score('train')
