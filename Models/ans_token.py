import theano
import theano.tensor as T
from theano.ifelse import ifelse
import lasagne
from Helpers import nn_utils


class AnsSelect(object):
    def __init__(self, inp_dim, hid_dim=50):

        self.dim = hid_dim
        self.inp_dim = inp_dim

        # Forming the input layer of the answer module
        q_sent_hid = T.vector("Question root node hidden State")
        ans_sent_hid = T.vector("Answer root node hidden state")
        ans_node_hid = T.vector("Answer word node hidden state")
        ans_parent_hid = T.vector("Answer word's parent hidden state")
        answer = T.scalar("Answer Probability")

        # Forming the processing layer
        self.W_q = nn_utils.normal_param(
            std=0.1, shape=(self.inp_dim, self.dim))
        self.W_ans_sent = nn_utils.normal_param(
            std=0.1, shape=(self.inp_dim, self.dim))
        self.W_ans_node = nn_utils.normal_param(
            std=0.1, shape=(self.inp_dim, self.dim))
        self.W_ans_parent = nn_utils.normal_param(
            std=0.1, shape=(self.inp_dim, self.dim))

        self.b_inp = nn_utils.constant_param(value=0.0, shape=(self.dim))

        self.W_hid = nn_utils.normal_param(
            std=0.1, shape=(self.dim))
        self.b_hid = nn_utils.constant_param(value=0.0, shape=())

        self.params = [self.W_q, self.W_ans_sent, self.W_ans_node,
                       self.W_ans_parent, self.b_inp, self.W_hid, self.b_hid]

        # Forming the output layer
        prediction = self.compute(
            q_sent_hid, ans_sent_hid, ans_node_hid, ans_parent_hid)

        # Forming the updates and loss layer
        loss = T.nnet.binary_crossentropy(prediction, answer)
        self.updates = lasagne.updates.adadelta(loss, self.params)

        self.train = theano.function([q_sent_hid, ans_sent_hid, ans_node_hid,
                                     ans_parent_hid, answer], [], updates=self.updates)

        self.predict = theano.function(
            [q_sent_hid, ans_sent_hid, ans_node_hid, ans_parent_hid], prediction)

        self.get_loss = theano.function([q_sent_hid, ans_sent_hid, ans_node_hid,
                                        ans_parent_hid, answer], loss)

    def compute(self, q_sent_hid, ans_sent_hid, ans_node_hid, ans_parent_hid):
        x = T.dot(q_sent_hid, self.W_q) + T.dot(ans_sent_hid, self.W_ans_sent) \
        + T.dot(ans_node_hid, self.W_ans_node) + \
            T.dot(ans_parent_hid, self.W_ans_parent)
        temp = T.nnet.sigmoid(x + self.b_inp)

        x = T.dot(temp, self.W_hid)
        out = T.nnet.sigmoid(x + self.b_hid)

        return out

    def save_params(self, file_name, epochs):
        with open(file_name, 'wb') as save_file:
            pickle.dump(
                obj={
                    'params': [x.get_value() for x in self.params],
                    'epoch': epochs,
                },
                file=save_file,
                protocol=-1
            )
        return

    def load_params(self, file_name):
        with open(file_name, 'rb') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)
        return
