'''
Attention-Based Bi CNN for Answer Sentence Selection from context.

'''

import tensorflow as tf
import numpy as np
import pickle
import sys
import operator
import time
import datetime
from tqdm import tqdm
from random import shuffle
import argparse

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

		ta = tf.TensorArray(dtype=tf.float32, size=self.max_sent_len)

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
		q_vector = self.attention_pooling(q_vector, attn_, 0)
		a_vector = self.attention_pooling(a_vector, attn_, 1)
		# q_vector = self.attention_pooling(q_vector, attn_)
		# a_vector = self.attention_pooling(a_vector, tf.transpose(attn_))


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


		# Padding before pooling
		q_vector = tf.pad(tf.squeeze(q_conv), padding)
		a_vector = tf.pad(tf.squeeze(a_conv), padding)
		# q_vector = tf.pad(tf.reduce_sum(tf.squeeze(q_conv), 2), padding)
		# a_vector = tf.pad(tf.reduce_sum(tf.squeeze(a_conv), 2), padding)

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
		# output_layer += 1e-7

		output_layer_test = tf.sigmoid(output_layer)

		# cross entropy loss
		loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=output_layer)
		# loss = -((self.label * tf.log(output_layer)) + ((1 - self.label) *
                                                 # tf.log(1 - output_layer)))

		# correct = tf.equal(self.label, output_layer)
		# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		train_step = optimizer.minimize(loss)

		return train_step, loss, output_layer_test


	def qa_vectorize(self, q, a, glove):
		
		# glove = utils.load_glove(200)

		q_vector = utils.get_vector_sequence(q, glove, 200)
		a_vector = utils.get_vector_sequence(a, glove, 200)
		q_vector = utils.pad_matrix_with_zeros(q_vector,
		                                       self.max_sent_len - len(q.split()))
		a_vector = utils.pad_matrix_with_zeros(a_vector,
		                                       self.max_sent_len - len(a.split()))
		# print(q_vector.shape, a_vector.shape)
		# print(" > Vectors Padded")
		return q_vector, a_vector


	def extract_features(self, q, a_list):

		a_list = [str(a) for a in a_list]

		tfidf, imp_tokens = infoRX.tf_idf(a_list, q)
		word_cnt = []

		for a in a_list:
			w_cnt = 0
			for imp in imp_tokens:
				if imp in a:
					w_cnt += 1
			word_cnt.append(w_cnt)

		return tfidf, word_cnt


	def model_state_saver(self, index, mode, u_dataset):
		filename = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'states/abcnn/state_')
		timestamp = ''
		
		if mode == 'train': # Saving model state at training completion
			timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

		else: # Saving model state at checkpoint
			timestamp = 'temp'
		
		with open(filename + 'r.txt', 'r') as fp:
			_, i, _ = tuple(fp.read().split(sep='\t'))
			index += int(i)

		with open(filename + 'r.txt', 'w') as fp:
			s = timestamp + '\t' + str(index) + '\t' + u_dataset
			fp.write(s)

		file_path = filename + timestamp + '.ckpt'
		print('\n> Model state will be saved @', file_path)

		return file_path # file_path is the path where the most recent training state will be saved.


	def model_state_loader(self):

		filename = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'states/abcnn/state_')
		index, u_dataset = None, None
		try:
			with open(filename + 'r.txt', 'r') as fp:
				timestamp, index, u_dataset = tuple(fp.read().split(sep='\t'))
				index = int(index)
				filename = filename + timestamp + '.ckpt'
				print('> Model will be restored from @', filename, index, u_dataset)
		except:
			pass

		return filename, index, u_dataset


	def run_model(self, mode):

		mark_init = time.time()
		score = 0
		p_score = 0
		p_instances = 0
		instances = 0
		pred_labl = -1
		one = tf.constant(1, dtype=tf.float32)
		train_step, loss, output_layer_test = self.model()
		saver = tf.train.Saver()
		

		q_list, a_list = utils._process_wikiqa_dataset(mode, self.max_sent_len)

		print(" > Dataset initialized. | Elapsed:", time.time() - mark_init)
		# q_list = q_list[:20]
		# a_list = a_list[:20]

		with tf.Session() as sessn:

			sessn.run(tf.global_variables_initializer())

			try:
				file_path, q_number = self.model_state_loader()
				saver.restore(sessn, file_path) 
				q_list = q_list[q_number:]
				a_list = a_list[q_number:]

			except:
				pass

			for i in tqdm(range(len(q_list)), total=len(q_list), ncols=75, unit='Question'):
				q = q_list[i]
				ans_list = a_list[i].keys()
				tfidf, word_cnt = self.extract_features(q, ans_list)
				j = 0

				for a in tqdm(ans_list, total=len(ans_list), ncols=75, unit='Answer'):

					mark_start = time.time()
					instances += 1
					label_ = a_list[i][a] # 0 if not answer and 1 if is answer

					q_vector, a_vector = self.qa_vectorize(q, a)

					input_dict = {self.q: q_vector, self.a: a_vector, self.label: label_,
					              self.word_cnt: word_cnt[j], self.tfidf: tfidf[j]}

					if mode == 'train':
						_, l, self.predict_label = sessn.run([train_step, loss, output_layer_test], feed_dict=input_dict)

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
						file_path = self.model_state_saver(i, 1)
						saver.save(sessn, file_path)

					accuracy = (score / instances) * 100 
					if p_instances > 0:
						p_accuracy = (p_score / p_instances) * 100 

					score, instances = 0, 0
					p_score, p_instances = 0, 0

					with open('result_abcnn.txt', 'a') as f:
						itr_res = str('> Accuracy: {0:.2f}'.format(accuracy) + '\n')
						if p_instances > 0:
							itr_res += str('> True_P accuracy: {0:.2f}'.format(p_accuracy) + '\n')
						f.write(itr_res)


			if mode == 'train':
				file_path = self.model_state_saver(0, 0)
				saver.save(sessn, file_path)

				
	def finalize_data(self, mode, u_dataset, babi_id):
		final_data = []
		# score = 0
		# instances = 0

		# p_score = 0
		# p_instances = 0
		
		# pred_labl = -1
		
		# mark_init = time.time()

		# saver = tf.train.Saver()
		# train_step, output_layer, loss, output_layer_test = self.model()

		dataset = []
		if u_dataset == 'wikiqa':
			print('> Getting dataset: {} {}'.format(u_dataset, mode))
			dataset = utils.get_wikiqa_for_abcnn(mode)
		else:
			print('> Getting dataset: {} {} {}'.format(u_dataset, mode, babi_id))
			dataset = utils.get_babi_for_abcnn(babi_id, mode)

		glove = utils.load_glove(200)
		print('> Vectorizing the questions and answers')
		for data in tqdm(dataset, total=len(dataset), ncols=75, unit='Pairs'):
			q, a, label, tfidf, word_cnt = data
			q_vector, a_vector = self.qa_vectorize(q, a, glove)
			final_data.append((q_vector, a_vector, label, tfidf, word_cnt))

		return final_data


	def run_model_v2(self, mode, u_dataset, babi_id):
		dataset = self.finalize_data(mode, u_dataset, babi_id)
		# print(dataset[0])
		mark_init = time.time()
		
		score = 0
		p_score = 0
		pred_pos = 0
		
		instances = 0
		p_instances = 0

		iteration = 0
		# result_file, accuracy_file = '', ''
		
		pred_labl = -1
		
		train_step, loss, output_layer_test = self.model()
		saver = tf.train.Saver()

		with tf.Session() as sess:

			if mode == 'train':
				filename, index, _u_dataset = self.model_state_loader()
				try:
					saver.restore(sess, filename)
					print(' > Model state restored from @ ' + filename)
					if _u_dataset == u_dataset:
						dataset = dataset[index: ]
						print('> Resuming training from instance {}'.format(index))
				except:
					sess.run(tf.global_variables_initializer())

			else:
				filename, index, _u_dataset = self.model_state_loader()
				try:
					saver.restore(sess, filename)
					print(' > Model state restored from @ ' + filename)
					if _u_dataset == u_dataset:
						dataset = dataset[index: ]
						print('> Resuming testing from instance {}'.format(index))
				except:
					print('> No saved state found. Exiting...')
					sess.close()
					import sys
					sys.exit()

			shuffle(dataset)

			for data in tqdm(dataset, total=len(dataset), ncols=75, unit=' QA Pairs'):
				mark_start = time.time()
				q_vector, a_vector, label, tfidf, word_cnt = data
				input_dict = {self.q: q_vector, self.a: a_vector, self.label: label, self.word_cnt: word_cnt, self.tfidf: tfidf}

				if mode == 'train':
					_, l, self.predict_label = sess.run([train_step, loss, output_layer_test], feed_dict=input_dict)
				else:
					l, self.predict_label = sess.run([loss, output_layer_test], feed_dict=input_dict)

				iteration += 1
				
				if self.predict_label[0] > 0.5:
					pred_labl = 1
					pred_pos += 1
				else:
					pred_labl = 0

				if label == 1:
					p_instances += 1
				instances += 1

				if pred_labl == label:
					score += 1
					if label == 1:
						p_score += 1

				with open('result_abcnn.txt', 'a') as f:
					itr_res = str('> QA' + str(iteration) + ' | Output Layer: ' + str(self.predict_label) + ' | Predicted Label: ' + str(pred_labl) + ' | Label: ' + str(label)+ ' | Loss:' + str(l)  + '| Elapsed: {0:.2f}'.format(time.time() - mark_start) + '\n')
					f.write(itr_res)

				if instances % 100 == 0:
					
					with open('accuracy_abcnn.txt', 'a') as f:
						
						itr_res = 'Iteration: {}\n'.format(iteration)

						accuracy = (score / instances) * 100
						itr_res += ' > Accuracy: {0:.2f}\n'.format(accuracy)

						if p_instances > 0:
							recall = (p_score / p_instances) * 100 
							itr_res += '  > True_+ve accuracy (recall): {0:.2f}\n'.format(recall)

						if pred_pos > 0:
							precision = (p_score / pred_pos) * 100 
							itr_res += '  > Pred_+ve accuracy (precision): {0:.2f}\n'.format(precision)

						f1 = 2 * precision * recall / (precision + recall) 
						itr_res += '  > F1 Score: {0:.2f}\n'.format(f1)

						f.write(itr_res)

					score, instances = 0, 0
					p_score, p_instances, pred_pos = 0, 0, 0

					if mode == 'train':
						file_path = self.model_state_saver(iteration, 'temp', u_dataset)
						saver.save(sess, file_path)
						print(' > Model state saved @ ' + file_path)

			if mode == 'train':
				file_path = self.model_state_saver(0, mode, u_dataset)
				saver.save(sess, file_path)
				print(' > Model state saved @ ' + file_path)


	def ans_select(self, question, ans_list):

		ans_sents = []

		tfidf, word_cnt = self.extract_features(q, ans_list)
		_, _, output_layer_test = self.model()

		with tf.Session() as sessn:
			
			filename, _, _ = self.model_state_loader()
			try:
				saver.restore(sess, filename)
				print(' > Model state restored from @ ' + filename)
			except:
				print(' > No saved state found. Exiting')
				sessn.close()
				sys.exit()

			glove = utils.load_glove(200)

			for i, ans in enumerate(ans_list):
			
				q_vector, a_vector = self.qa_vectorize(question, ans, glove)

				input_dict = {self.q: q_vector, self.a: a_vector, self.label: None, self.word_cnt: word_cnt[i], self.tfidf: tfidf[i]}
				pred = sessn.run(output_layer_test, feed_dict=input_dict)

				ans_sents.append((ans, pred))

		ans_sents = sorted(ans_sents, key=operator.itemgetter(1), reverse=True) # Sorts by scores in desc order
		
		return ans_sents


# model verification
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, metavar='mode', default='train', help='What mode to run the model in: train(default), test')
	parser.add_argument('--dataset', type=str, metavar='u_dataset', default='babi', help='Select the dataset to use: babi(default), wikiqa')
	parser.add_argument('--babi_id', type=str, metavar='babi_id', default='1', help='Select which babi set to use: 1(default) - 20')
	parser.set_defaults(shuffle=True)

	args = parser.parse_args()
	mode, u_dataset, babi_id = args.mode, args.dataset, args.babi_id

	selector = abcnn_model()
	selector.run_model_v2(mode=mode, u_dataset=u_dataset, babi_id=babi_id)
