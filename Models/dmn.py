
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
    def __init__(self, babi_train_raw, babi_test_raw,babi_deploy_raw, word2vec, word_vector_size,
                dim, mode, answer_module, input_mask_mode, memory_hops, l2,
                normalize_attention, answer_vec, debug,sentEmbdType="basic",**kwargs):

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
        if(self.sentEmbdType != "basic"):
            self.dep_tags_dict=utils.load_dep_tags()
            import spacy
            self.nlp=spacy.load('en')

        if self.mode != 'deploy':
            print("==> not used params in DMN class:", kwargs.keys())

            self.train_input, self.train_q, self.train_answer, self.train_input_mask, self.train_input_dep_tags, self.train_q_dep_tags = self._process_input(babi_train_raw,self.sentEmbdType)
            self.test_input, self.test_q, self.test_answer, self.test_input_mask, self.test_input_dep_tags, self.test_q_dep_tags = self._process_input(babi_test_raw,self.sentEmbdType)
        else:
            self.deploy_input, self.deploy_q, self.deploy_answer, self.deploy_mask, self.deploy_input_dep_tags, self.deploy_q_dep_tags = self._process_input(babi_deploy_raw,self.sentEmbdType)

        self.vocab_size = len(self.vocab)


        if self.debug:
            print('Input:',np.array(self.train_input).shape)
            print('Quest:',np.array(self.train_q).shape)
            print('Answer:',np.array(self.train_answer).shape)
            print('Mask:',np.array(self.train_input_mask))
            sys.exit(0)

        if self.answer_vec == 'word2vec':
            self.answer_var = T.vector('answer_var')
        else:
            self.answer_var = T.iscalar('answer_var')

        if self.answer_vec == 'one_hot' or self.answer_vec == 'index':
            self.answer_size = self.vocab_size
        elif self.answer_vec == 'word2vec':
            self.answer_size = self.word_vector_size
        else:
            raise Exception("Invalid answer_vec type")

        if self.mode != 'deploy': print("==> building input module")


        if self.mode != 'deploy': print("==> creating parameters for memory module")
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


        if self.mode != 'deploy': print("==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops)
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(self.memory[iter - 1])
            self.memory.append(self.GRU_update(self.memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))

        self.last_mem = self.memory[-1]

        if self.mode != 'deploy': print("==> building answer module")

        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.answer_size, self.dim))

        if self.answer_module == 'feedforward':
            self.prediction = nn_utils.softmax(T.dot(self.W_a, self.last_mem))

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
        #         outputs_info=[self.last_mem, T.zeros_like(dummy)],
        #         n_steps=1)
        #     self.prediction = results[1][-1]

        else:
            raise Exception("invalid answer_module")


        if self.mode != 'deploy': print("==> collecting all parameters")
        self.params = [self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid,
                  self.W_b, self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]

        if self.answer_module == 'recurrent':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res,
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]


        if self.mode != 'deploy': print("==> building loss layer and computing updates")
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

    def get_SentenceVecs(self,sentences,tags=None):
        # print(np.array(sentences).shape)
        if self.sentEmbdType == 'basic':
            sentVecs=self.sent_embd.predict(np.array(sentences))
        elif self.sentEmbdType == 'advanced':
            sentVecs=self.sent_embd.predict(np.array(sentences),np.array(tags).reshape(-1))
        return sentVecs


    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper:
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
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
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0]))

        if (self.normalize_attention):
            g = nn_utils.softmax(g)

        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))

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


    def _process_input(self, data_raw, sentEmbdType):
        questions = []
        inputs = []
        answers = []
        input_masks = []
        dep_tags_inputs=[]
        dep_tags_q=[]
        for x in data_raw:
            inp = x["C"].lower().split(' ')
            inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]

            inp_vector = [utils.process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.word_vector_size,
                                        to_return = "word2vec") for w in inp]

            q_vector = [utils.process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.word_vector_size,
                                        to_return = "word2vec") for w in q]

            if(sentEmbdType!="basic"):
                # print(x["C"])
                # print(x["Q"])
                dep_tags_inp_vector = utils.get_depTags_sequence(x["C"], self.dep_tags_dict, self.nlp)
                dep_tags_q_vector = utils.get_depTags_sequence(x["Q"], self.dep_tags_dict, self.nlp)
                dep_tags_inputs.append(np.vstack(dep_tags_inp_vector))
                dep_tags_q.append(np.vstack(dep_tags_q_vector))


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

        return inputs, questions, answers, input_masks,dep_tags_inputs,dep_tags_q


    def get_batches_per_epoch(self, mode):
        if mode == 'train':
            return len(self.train_input)
        elif mode == 'test':
            return len(self.test_input)
        else:
            raise Exception("unknown mode")


    def shuffle_train_set(self):
        print("==> Shuffling the train set")
        combined = list(zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask))
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = zip(*combined)


    def get_step_inputs(self, batch_index, mode):
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")

        sent_dep_tags=None
        q_dep_tags=None

        if mode == "train":
            theano_fn = self.train_fn
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            input_masks = self.train_input_mask
            if(self.sentEmbdType=='advanced'):
                sent_dep_tags=self.train_input_dep_tags[batch_index]
                q_dep_tags=self.train_q_dep_tags[batch_index]
        elif mode == "test":
            theano_fn = self.test_fn
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
            if(self.sentEmbdType=='advanced'):
                sent_dep_tags=self.test_input_dep_tags[batch_index]
                q_dep_tags=self.test_q_dep_tags[batch_index]
        else:
            raise Exception("Invalid mode")

        inp = inputs[batch_index]
        q = qs[batch_index]
        ans = answers[batch_index]
        input_mask = input_masks[batch_index]


        return theano_fn,inp,q,ans,input_mask,sent_dep_tags,q_dep_tags

        # skipped = 0
        # grad_norm = float('NaN')

        # if mode == 'train':
        #     gradient_value = self.get_gradient_fn(inp, q, ans, input_mask)
        #     grad_norm = np.max([utils.get_norm(x) for x in gradient_value])

        #     if (np.isnan(grad_norm)):
        #         print("==> gradient is nan at index %d." % batch_index)
        #         print("==> skipping")
        #         skipped = 1

        # if skipped == 0:
        #     ret = theano_fn(inp, q, ans, input_mask)
        # else:
        #     ret = [-1, -1]

        # param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])

        # return {"prediction": np.array([ret[0]]),
        #         "answers": np.array([ans]),
        #         "current_loss": ret[1],
        #         "skipped": skipped,
        #         "log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
        #         }
    def step_deploy(self):
        inputs = self.train_input
        q = self.train_q
        input_mask = self.train_input_mask
        prediction = self.deploy_fn(inputs[0],q[0],input_mask[0])
        return np.array([prediction])

    def generate_functions(self,input_list,output_list=[]):
        if self.mode == 'deploy':
            self.deploy_fn = theano.function(inputs=input_list,outputs=[self.prediction],on_unused_input='ignore')

        else:
            updates = lasagne.updates.adadelta(self.loss, self.params)

            if self.mode == 'train':
                print("==> compiling train_fn")
                self.train_fn = theano.function(inputs=input_list,
                                           outputs=[self.prediction, self.loss],
                                           updates=updates)

            print("==> compiling test_fn")
            self.test_fn = theano.function(inputs=input_list,
                                      outputs=[self.prediction, self.loss, self.last_mem]+output_list)


            if self.mode == 'train':
                print("==> computing gradients (for debugging)")
                gradient = T.grad(self.loss, self.params)
                self.get_gradient_fn = theano.function(inputs=input_list, outputs=gradient)


class DMN_basic(DMN):
    def __init__(self,**kwargs):

        self.dim = kwargs['dim']
        self.word_vector_size = kwargs['word_vector_size']

        self.input_var = T.matrix('input_var')
        self.q_var = T.matrix('question_var')
        self.input_mask_var = T.ivector('input_mask_var')

        # Setting up Sentence Embedder Module
        self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        inp_c_history, _ = theano.scan(fn=self.input_gru_step,
                    sequences=self.input_var,
                    outputs_info=T.zeros_like(self.b_inp_hid))

        self.inp_c = inp_c_history.take(self.input_mask_var, axis=0)

        self.q_q, _ = theano.scan(fn=self.input_gru_step,
                    sequences=self.q_var,
                    outputs_info=T.zeros_like(self.b_inp_hid))

        self.q_q = self.q_q[-1]

        self.memory = [self.q_q.copy()]

        super().__init__(**kwargs)

        self.params += [self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res,
                  self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                  self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid]


        input_list = [self.input_var, self.q_var, self.answer_var, self.input_mask_var]
        output_list = [self.inp_c,self.q_q]

        self.generate_functions(input_list,output_list)


    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res,
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    def step(self, batch_index, mode):
        theano_fn,inp,q,ans,input_mask,_,_=self.get_step_inputs(batch_index,mode)
        ret = theano_fn(inp, q, ans, input_mask)
        return {"prediction": np.array([ret[0]]),
                "answers": np.array([ans]),
                "current_loss": ret[1],
                # "skipped": skipped,
                # "log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
                }


class DMN_Erudite(DMN):
    def __init__(self,**kwargs):
        self.inp_c = T.matrix('inp')
        self.q_q = T.vector('q_q')
        
        self.memory = [self.q_q.copy()]
        
        super().__init__(**kwargs)

        #Setting up pre-trained Sentence Embedder for question and input module:
        if self.mode != 'deploy': print("==> Setting up pre-trained Sentence Embedder")       
        if self.sentEmbdType=="basic":
            self.sent_embd=SentEmbd.SentEmbd_basic(self.word_vector_size,self.dim)
        else:
            self.sent_embd=SentEmbd.SentEmbd_syntactic(self.word_vector_size,self.dim,len(self.dep_tags_dict))

        if(self.mode == 'deploy'):
            input_list = [self.inp_c,self.q_q]
        else:
            input_list = [self.inp_c,self.q_q,self.answer_var]
        
        self.generate_functions(input_list)

    def step_deploy(self):
        
        inputs=self.deploy_input
        q=self.deploy_q
        input_mask=self.deploy_mask
        # print(self.deploy_input_dep_tags[0])
        # print(self.deploy_q_dep_tags[0])
        
        if(self.sentEmbdType=='basic'):
            inp_vec=self.get_SentenceVecs(inputs[0])
            q_vec=self.get_SentenceVecs(q[0])
        elif(self.sentEmbdType=='advanced'):
            inp_vec=self.get_SentenceVecs(inputs[0],self.deploy_input_dep_tags[0])
            q_vec=self.get_SentenceVecs(q[0],self.deploy_q_dep_tags[0])

        inp_vec=inp_vec.take(input_mask, axis=0)
        q_vec=q_vec[-1]
        
        prediction = self.deploy_fn(inp_vec[0],q_vec)
        ans_ind=np.argmax(prediction)
        
        print(ans_ind)
        print(self.ivocab)
        print(self.ivocab[ans_ind])
        
        return prediction

    def step(self, batch_index, mode):
        theano_fn,inp,q,ans,input_mask,sent_dep_tags,q_dep_tags=self.get_step_inputs(batch_index,mode)
        
        inp_vec=self.get_SentenceVecs(inp,sent_dep_tags)
        inp_vec=inp_vec.take(input_mask, axis=0)
        
        q_vec=self.get_SentenceVecs(q,q_dep_tags)
        q_vec=q_vec[-1]

        ret = theano_fn(inp_vec, q_vec, ans)

        return {"prediction": np.array([ret[0]]),
                "answers": np.array([ans]),
                "current_loss": ret[1],
                # "skipped": skipped,
                # "log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
                }

