'''
Attention-Based Bi CNN for Answer Sentence Selection from context.

'''

import tensorflow as tf
import numpy as np
import csv
import glove_utils as glove
from Helpers import utils
from nltk.tokenize.moses import MosesTokenizer

tokenizer = MosesTokenizer()
glove_wordmap = glove.get_glove()


def sentence2sequence(sentence):
	tokens = tokenizer.tokenize(sentence.lower())
	rows = []
	words = []

	for token in tokens:
		i = len(token)
		while len(token) > 0:
			word = token[:i]
			if word in glove_wordmap:
				rows.append(glove_wordmap[word])
				words.append(word)
				token = token[i:]
				i = len(token)
				continue
			else:
				i -= 1
			if i == 0:
				rows.append(glove.fill_unknown(token))
				words.append(token)
				break
	return np.array(rows), words


def _process_dataset(self, file):
	questions = []
	answers = []
	file = ".\Dataset\WikiQACorpus\WikiQA-dev.tsv"

	with open(file, encoding="utf8") as data_file:
		source = list(csv.reader(data_file, delimiter="\t", quotechar='"'))
		q_index = 'Q-1'
		ans_sents = []

		for row in source[1:]:

			if q_index != row[0]:
				answers.append(ans_sents)
				ans_sents = []
				questions.append(row[1])
				q_index = row[0]

		ans_sents.append({row[5]: row[6]})

	answers.append(ans_sents)
	answers = answers[1:]

	for i in range(len(questions)):
		print("Question:", questions[i])
		print("Answers:", answers[i])

	return questions, answers


# Hyperparameters

sequence_length  # no of tokens
# no_classes #output label classes

d = 200  # vector_dim
s = 40 #max_sent_length
h = 4  # filter_size
m = 50  # num_filters
alpha = 0.05 # learning_rate

# Q = tf.placeholder(tf.float32, [None, sequence_length, 50])
# A = tf.placeholder(tf.float32, [None, )

# Y_ = tf.placeholder(tf.float32, [None, 10])

W_q = tf.Variable(tf.random_normal([]))
W_a = tf.Variable(tf.random_normal([]))


def abcnn_model(input_data, mode):

	q_set = input_data["Q"]
	a_set = input_data["A"]

	for i in range(len(q_set)):
		q_vector, _ = sentence2sequence(q_set[i])

		# for a in a_set[i]:
		a_vector, _ = sentence2sequence(a_set[i][0])

		attn_ = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(q_vector, a_vector)), reduction_indices=1))

		q_feature_ = tf.matmul(attn_, W_q) 
		a_feature_ = tf.matmul(attn_, W_a, transpose_a=True)	

		q_vector = tf.reshape([q_feature_, q_vector], [None, d, 2])
		a_vector = tf.reshape([a_feature_, a_vector], [N])



		# Convolutional Layer
		q_conv = tf.layers.conv2d(
			inputs=q_vector,
			# filters=m,
			kernel_size=[h, d],
			padding="same",
			activation=tf.nn.relu)

		a_conv = tf.layers.conv2d(
			inputs=a_vector,
			# filters=m,
			kernel_size=[h, d],
			padding="same",
			activation=tf.nn.relu)

		# TODO: Define pool sizes and strides
		pool1 = tf.layers.average_pooling2d(inputs=q_conv, pool_size=[2, 2], strides=2)
		pool2 = tf.layers.average_pooling2d(inputs=a_conv, pool_size=[2, 2], strides=2)

		pool_concat = tf.concat([pool1, pool2], 0)
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
		sessn.run(tf.global_variables_initializer())

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
a, c = sessn.run([accuracy, cross_entropy], feed_dict=test_data)
print('Test Accuracy : ', a)

if __name__ == '__main__':
	train_neural_net(X)
