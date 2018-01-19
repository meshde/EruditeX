'''
Attention-Based Bi CNN for Answer Sentence Selection from context.

'''

import tensorflow as tf
import numpy as np
from Helpers import utils

class abcnn_model:

	def __init__(self, input_data, mode):

		# Hyperparameters

		self.vector_dim = 200  # vector_dim
		self.max_sent_len = 40  # max_sent_length
		self.filter_size = 4  # filter_size
		self.n_filters = 50  # num_filters
		self.learning_rate = 0.05  # learning_rate

		self.q = tf.placeholder(tf.float32, [self.vector_dim, self.max_sent_len], 'question')
		self.a = tf.placeholder(tf.float32, [self.vector_dim, self.max_sent_len], 'answer')

		self.W_q = tf.Variable(tf.random_normal([self.vector_dim, self.max_sent_len]))
		self.W_a = tf.Variable(tf.random_normal([self.vector_dim, self.max_sent_len]))

	
	def attention_pooling(feature_map, attn_):

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


	def model(q_vector, a_vector):

		attn_ = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(q_vector, a_vector)), reduction_indices=1))

		q_feature_ = tf.matmul(attn_, W_q)
		a_feature_ = tf.matmul(attn_, W_a, transpose_a=True)

		q_vector = tf.stack([q_feature_, q_vector])
		a_vector = tf.stack([a_feature_, a_vector])
		
		# Convolutional Layer1
		q_conv1 = tf.layers.conv2d(
			inputs=q_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

		a_conv2 = tf.layers.conv2d(
			inputs=a_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

		attn_ = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(q_vector, a_vector)), reduction_indices=1))
	
		q_vector = attention_pooling(q_vector, attn_)
		a_vector = attention_pooling(a_vector, tf.transpose(attn_))


		# Convolutional Layer2
		q_conv2 = tf.layers.conv2d(
			inputs=q_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

		a_conv2 = tf.layers.conv2d(
			inputs=a_vector,
			filters=self.n_filters,
			kernel_size=[self.filter_size, self.vector_dim],
			padding="same",
			activation=tf.nn.relu)

			
		dense = tf.layers.dense(inputs=pool_concat, units=1024, activation=tf.nn.relu)

		dropout = tf.layers.dropout(
			inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
		# Logits Layer
		logits = tf.layers.dense(inputs=dropout, units=10)

		predictions = {
			"classes": tf.argmax(input=logits, axis=1),
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
		loss = tf.losses.softmax_cross_entropy(
			onehot_labels=onehot_labels, logits=logits)

		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Add evaluation metrics (for EVAL mode)
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=predictions["classes"])}
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
		

	def get_score(self, q_vector, q_len, a_vector, a_len):

		# s = max(q_len, a_len)
		q_vector = utils.pad_matrix_with_zeros(q_vector, 40 - q_len)
		a_vector = utils.pad_matrix_with_zeros(a_vector, 40 - a_len)

		input_dict = {q: q_vector, a: a_vector}
		score = -1

		optimizer = tf.train.Adamoptimizer(self.learning_rate)
		# TODO: define loss
		train_step = optimizer.minimize(loss)

		Y = self.model(self.q, self.a)

		with tf.Session() as sessn:
			sessn.run(tf.global_variables_initializer())
			score = sessn.run(train_step, feed_dict=input_dict)

		return score

	def train_neural_net(X):
		Y = abcnn_model(X)

		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_)  # loss function
		cross_entropy = tf.reduce_mean(cross_entropy) * 100

		correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		# Training Step

		optimizer = tf.train.GradientDescentOptimizer(0.003)
		train_step = optimizer.minimize(cross_entropy)

		no_epochs = 1

		with tf.Session() as sessn:
			# Training Stage
			for epoch in range(no_epochs):
				epoch_loss = 0

				batch_X = np.reshape(batch_X, (-1, 28, 28, 1))
				train_dict = {X: batch_X, Y_: batch_Y}

				sessn.run(train_step, feed_dict=train_dict)
				a, c = sessn.run([accuracy, cross_entropy], feed_dict=train_dict)
				# print('Batch',i,' / Cost :', c, ' / Accuracy :', a)
				epoch_loss += c

			print('Finished Epoch', epoch, '> loss : ', epoch_loss)

test_data = {X: np.reshape(
	a, c=sessn.run([accuracy, cross_entropy], feed_dict=test_data))
	print('Test Accuracy : ', a)


# model verification
if __name__ == '__main__':

	q_list, a_list = utils._process_wikiqa_dataset("..\data\wikiqa\WikiQA-train.tsv")

	for i in range(len(q_list)):
		q = q_list[i]
		for a in a_list[i]:
			result = abcnn_model.get_score(utils.get_vector_sequence(q, utils.load_glove()), len(q.split()),
		                               utils.get_vector_sequence(a, utils.load_glove()), len(a.split()))

	print(i + " : " + result)

	# 	print("Question:", q_list[i])
	# 	print("Answers:", a_list[i])


	# q_list, a_list = utils._process_wikiqa_dataset("..\data\wikiqa\WikiQA-test.tsv")
