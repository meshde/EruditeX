import tensorflow as tf

class ans_node(object):
	def __init__(self, dim):
		# Hyperparameters
		self.dim = dim
		self.n_features = 4
		self.learning_rate = 0.0005

		# Placeholders
		self.a = tf.placeholder(tf.float32, [self.dim], 'answer_node')
		self.a_p = tf.placeholder(tf.float32, [self.dim], 'parent_node')
		self.a_root = tf.placeholder(tf.float32, [self.dim], 'answer_root')
		self.q = tf.placeholder(tf.float32, [self.dim], 'question_root')
		self.label = tf.placeholder(tf.float32, 'correct_ans_label')

		# Weights and biases
		self.W_1 = tf.variable(tf.random_normal([self.dim]))
		self.b_1 = tf.variable(tf.random_normal([self.dim, 1]))

		self.W_2 = tf.variable(tf.random_normal([self.n_features, 1]))
		self.b_2 = tf.variable(tf.random_normal([self.n_features, 1]))

	def model(self):
		inp = tf.stack([self.a, self.a_p, self.q])
		_res = tf.add(tf.matmul(inp, self.W_1), self.b_1)
		# _res = tf.transpose(_res)
		res = tf.add(tf.matmul(_res, self.W_2, transpose_a=True), self.b_2)
		# prob = tf.nn.sigmoid(res)
		# self.corr = tf.cond(tf.less(prob, self.corr_threshold), true_fn=False, false_fn=True)
		# The above function will return True if node is answer and vice versa

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.corr, logits=res, name='loss_function')
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		step = optimizer.minimize(loss)

		return step

	def run_model(self):
		train_step = self.nn_model()
		dataset = self.get_dataset()

		with tf.session() as sess:
			sess.run(tf.global_variables_initializer())
			for data in dataset:
				sess.run([train_step], feed_dict=data)



