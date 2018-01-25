'''
Attention-Based Bi CNN for Answer Sentence Selection from context.

'''

import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')

from Helpers import utils
from IR import infoRX
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics.pairwise import euclidean_distances
# from scipy.spatial.distance import cdist


class abcnn_model:

	def __init__(self):

		# Hyperparameters

		self.vector_dim = 200  # vector_dim
		self.max_sent_len = 40  # max_sent_length
		self.filter_size = 4  # filter_size
		self.n_filters = 50  # num_filters
		self.learning_rate = 0.05  # learning_rate

		self.q = tf.placeholder(tf.float64, [self.max_sent_len, self.vector_dim], 'question')
		self.a = tf.placeholder(tf.float64, [self.max_sent_len, self.vector_dim], 'answer')

		self.Q_ = tf.placeholder(tf.float64, [self.vector_dim])
		self.A_ = tf.placeholder(tf.float64, [self.vector_dim])

		# [self.max_sent_len, self.vector_dim] [self.vector_dim, self.max_sent_len]
		self.W_q = tf.Variable(tf.random_normal([self.max_sent_len, self.vector_dim] ))
		self.W_a = tf.Variable(tf.random_normal([self.max_sent_len, self.vector_dim] ))

		self.W_lr = tf.Variable(tf.random_normal([3]))
		self.B_lr = tf.Variable(tf.random_normal([3]))

	def pairwise_euclidean_dist(self, m0, m1):

		ted = tf.TensorArray(dtype=tf.float32, size=self.max_sent_len)

		c1 = lambda i, j, m0, m1, ta: j < self.max_sent_len 
		def body1(i, j, m0, m1, ta):
			to = ta.write(j, tf.reciprocal(1+tf.sqrt(tf.reduce_sum(tf.square(m0[i] - m1[j])))))
			return i, j+1, m0, m1, to

		c0 = lambda i, m0, m1, ted: i < self.max_sent_len
		def body0(i, m0, m1, ted):
			ta = tf.TensorArray(dtype=tf.float32, size=self.max_sent_len)
			tb = tf.while_loop(c1, body1, loop_vars=[i, 0, m0, m1, ta])[-1]
			tc = tb.stack()
			ted = ted.write(i, tc)
			return i+1, m0, m1, ted

		ta_final = tf.while_loop(c0, body0, loop_vars=[0, m0, m1, ted])[-1]
		result = ta_final.stack()

		return result

	def attention_pooling(self, feature_map, attn_):

		attn_pool_mat = tf.placeholder(tf.float32, [None, None])

		for i in range(self.max_sent_len):
			# ind_attn_ft = tf.constant([i, i+self.filter_size])

			# r_feature = tf.gather(feature_map, ind_attn_ft)
			# r_attn = tf.gather(attn_, ind_attn_ft)

			temp_mat = tf.placeholder(tf.float32, [None])

			for j in range(i, i+self.filter_size):
			
				index = tf.constant([j])
				tf.concat(temp_mat, tf.matmul(tf.gather(feature_map, index), tf.gather(attn_, index)))

			tf.concat([attn_pool_mat, tf.reduce_sum(temp_mat, 1)], 0)

		return attn_pool_mat


	def model(self, q_vector, a_vector):

		attn_ = self.pairwise_euclidean_dist(q_vector, a_vector)
		# attn_ = tf.cast(euclidean_distances(q_vector, a_vector), tf.float32)
		# attn_ = tf.cast( tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(q_vector, a_vector)), 1)), tf.float32)
		# attn_ = cdist(q_vector, a_vector)



		# print(q_vector.get_shape(), a_vector.get_shape())
		print(type(q_vector), type(a_vector))
		print(attn_.get_shape(), self.W_q.get_shape(), self.W_a.get_shape())

		q_feature_ = tf.matmul(attn_, self.W_q)
		a_feature_ = tf.matmul(attn_, self.W_a, transpose_a=True)

		q_vector = tf.stack([q_feature_, q_vector])
		a_vector = tf.stack([a_feature_, a_vector])

		# Expanding dimensions for conv input
		q_vector = tf.reshape(q_vector, [-1, 2, self.max_sent_len, self.vector_dim, 1])
		a_vector = tf.reshape(a_vector, [-1, 2, self.max_sent_len, self.vector_dim, 1])

		# q_vector = tf.expand_dims(tf.expand_dims(q_vector, 3), 0)
		# a_vector = tf.expand_dims(tf.expand_dims(a_vector, 3), 0)

		# Convolutional Layer1
		q_conv1 = tf.layers.conv3d(
			inputs=q_vector,
			filters=self.n_filters,
			kernel_size=[2, self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

		a_conv1 = tf.layers.conv3d(
			inputs=a_vector,
			filters=self.n_filters,
			kernel_size=[2, self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

		print(q_conv1.get_shape(), a_conv1.get_shape())
		print(type(q_conv1), type(a_conv1))


		attn_ = self.pairwise_euclidean_dist(q_vector, a_vector)
		# attn_ = tf.cast(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(q_vector, a_vector)), reduction_indices=1)), tf.float32)
		# attn_ = cdist(q_vector, a_vector)

		q_vector = self.attention_pooling(q_vector, attn_)
		a_vector = self.attention_pooling(a_vector, tf.transpose(attn_))

		# Expanding dimensions for conv input
		q_vector = tf.expand_dims(tf.expand_dims(q_vector, 3), 0)
		a_vector = tf.expand_dims(tf.expand_dims(a_vector, 3), 0)

		# Convolutional Layer2
		q_conv2 = tf.layers.conv3d(
			inputs=q_vector,
			filters=self.n_filters,
			kernel_size=[2, self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

		a_conv2 = tf.layers.conv3d(
			inputs=a_vector,
			filters=self.n_filters,
			kernel_size=[2, self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

		attn_ = self.pairwise_euclidean_dist(q_vector, a_vector)
		# attn_ = tf.cast(euclidean_distances(q_conv2, a_conv2), tf.float32)
		# attn_ = cdist(q_vector, a_vector)

		q_vector = self.attention_pooling(q_vector, attn_)
		a_vector = self.attention_pooling(a_vector, tf.transpose(attn_))			
		
		pool1 = tf.layers.average_pooling2d(inputs=q_vector, pool_size=[self.vector_dim, self.max_sent_len])
		pool2 = tf.layers.average_pooling2d(inputs=a_vector, pool_size=[self.vector_dim, self.max_sent_len])

		return pool1, pool2


	def get_score(self, q_vector, a_vector, label, q_len, a_len, word_cnt, tfidf):

		# s = max(q_len, a_len)	

		# self.q = utils.pad_matrix_with_zeros(q_vector, self.max_sent_len - q_len)
		# self.a = utils.pad_matrix_with_zeros(a_vector, self.max_sent_len - a_len)
		
		q_vector = utils.pad_matrix_with_zeros(q_vector, self.max_sent_len - q_len)
		a_vector = utils.pad_matrix_with_zeros(a_vector, self.max_sent_len - a_len)
		print(" > Vectors Padded")

		input_dict = {self.q: q_vector, self.a: a_vector}
		score = -1

		optimizer = tf.train.AdamOptimizer(self.learning_rate)

		Q_, A_ = self.model(q_vector, a_vector)

		cos_sim = infoRX.cosine_similarity(Q_, A_)
		features = np.array([cos_sim, word_cnt, tfidf])

		output_layer = tf.nn.sigmoid(tf.add(tf.matmul(self.W_lr, features), self.B_lr))

		# cross entropy loss
		loss = -tf.reduce_sum(label * tf.log(output_layer))

		correct = tf.equal(label, output_layer)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		train_step = optimizer.minimize(loss)

		with tf.Session() as sessn:
			sessn.run(tf.global_variables_initializer())
			sessn.run(train_step, feed_dict=input_dict)
			score, l = sessn.run([output_layer, loss], feed_dict=input_dict)

		return score
 

# model verification
if __name__ == '__main__':

	selector = abcnn_model()
	glove = utils.load_glove(200)		

	q_list, a_list = utils._process_wikiqa_dataset("..\data\wikiqa\WikiQA-train.tsv")
	print(" > Dataset initialized.")

	for i in range(len(q_list)):
		q = q_list[i]
		tfidf = infoRX.tf_idf([str(a) for a in a_list[i].keys()], q)

		for a in a_list[i].keys():

			word_cnt = 0
			imp_tokens = [i for i in word_tokenize(q.lower()) if i not in set(stopwords.words('english'))]
			for x in imp_tokens:
				if x in a:
					word_cnt += 1

			result = selector.get_score(utils.get_vector_sequence(q, glove, 200) , utils.get_vector_sequence(a, glove, 200), a_list[i][a], len(q.split()), len(a.split()), word_cnt, tfidf)

		print(i + " : " + result)

	# 	print("Question:", q_list[i])
	# 	print("Answers:", a_list[i])


	# q_list, a_list = utils._process_wikiqa_dataset("..\data\wikiqa\WikiQA-test.tsv")
