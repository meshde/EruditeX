#Sentence Embedder.
from Helpers import utils
from Helpers import nn_utils
import numpy as np
import theano.tensor as T
import theano
import lasagne
import os
import sys
import pickle

class SentEmbd(object):
    def __init__(self,word_vector_size,dim,visualise=False):

        self.visualise = visualise

        self.dim=dim #Dimmensions of Hidden State of the GRU
        self.word_vector_size=word_vector_size
        self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.U_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.U_inp_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.U_inp_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self. dim))
        self.b_inp_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))


        self.similarity_score = T.dscalar('score')
        self.sent1=T.dmatrix('sent1')
        self.sent2=T.dmatrix('sent2')
        self.params = [
        self.W_inp_res_in,
        self.U_inp_res_hid ,
        self.b_inp_res ,

        self.W_inp_upd_in,
        self.U_inp_upd_hid,
        self.b_inp_upd,

        self.W_inp_hid_in,
        self.U_inp_hid_hid,
        self.b_inp_hid
]

        self.hid1=None
        self.hid2=None
        self.hid_state1=None
        self.hid_state2=None
        self.train=None
        self.get_similarity=None
        self.updates=None
        self.score=None
        self.predict=None



        # if self.visualise:
        #     self.get_loss = theano.function([sent1,sent2,similarity_score],[self.loss])
        #     self.get_updates = {}
        #     for param in self.params:
        #         self.get_updates[param] = theano.function([sent1,sent2,similarity_score],[lasagne.updates.adadelta(self.loss, [param])[param]])

    def generate_function(self):
        self.score = (((nn_utils.cosine_similarity(self.hid1,self.hid2) + 1)/2) * 4) + 1
        loss = T.sqrt(abs(T.square(self.score)-T.square(self.similarity_score)))
        # self.get_similarity = theano.function([self.sent1,self.sent2],[predicted_score])
        self.updates = lasagne.updates.adadelta(loss, self.params) #BlackBox

    def computation(self,wVec,prev_hid_state):
        zt=T.nnet.sigmoid(T.dot(self.W_inp_upd_in,wVec)+ T.dot(self.U_inp_upd_hid,prev_hid_state) + self.b_inp_upd)
        rt=T.nnet.sigmoid(T.dot(self.W_inp_res_in,wVec)+ T.dot(self.U_inp_res_hid,prev_hid_state) + self.b_inp_res)
        curr_hid_state_int=T.tanh(T.dot(self.W_inp_hid_in,wVec) + (rt * (T.dot(self.U_inp_hid_hid,prev_hid_state))) + self.b_inp_hid) # intermediate hidden state
        t=(zt * prev_hid_state)+((1-zt) * curr_hid_state_int) #Hidden state(ht) at timestamp t
        return t

    def trainx(self,training_dataset,exp_dataset,relatedness_scores,epochs):
        for val in range(epochs):
            for num in np.arange(len(training_dataset)):
                self.train(np.array(training_dataset[num]),np.array(exp_dataset[num]),relatedness_scores[num])


    def testing(self,training_dataset,exp_dataset,relatedness_scores,log_file,choice,additional_inputs=[]):

        avg_acc=0.0
        dep_tags1=[]
        dep_tags2=[]
        if(choice==2):
            dep_tags1=additional_inputs[0]
            dep_tags2=additional_inputs[1]
        with open(log_file,'w') as f:
            for num in np.arange(len(training_dataset)):
                if(choice==1):
                    score = self.get_similarity(np.array(training_dataset[num]),np.array(exp_dataset[num])) #For SentEmbd_Basic
                elif(choice==2):
                    score = self.get_similarity(np.array(training_dataset[num]),np.array(exp_dataset[num]),np.array(dep_tags1[num]),np.array(dep_tags2[num])) #For SentEmbd_Syntactic
                f.write("Actual Similarity: "+str(score)+"\n")
                f.write("Expected Similarity(From SICK.txt): "+str(relatedness_scores[num])+"\n")
                avg_acc += (abs(score[0]-relatedness_scores[num])/relatedness_scores[num])
            avg_acc =(avg_acc/len(training_dataset) * 100)
            f.write("Average Accuracy: "+str(avg_acc)+"\n")
        return avg_acc

    def printAllParams(self):
        for param in self.params:
            print(utils.get_var_name(param,self.__dict__))
            print(param.get_value())
        return

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

    def predictx(self,inp_sent,glove):
        vectorizedSent=utils.get_vector_sequence(inp_sent.strip(),glove,self.word_vector_size)
        sentVector=self.predict(vectorizedSent)
        return sentVector


class SentEmbd_basic(SentEmbd):
    def __init__(self,word_vector_size,dim,visualise=False):
        super().__init__(word_vector_size,dim,visualise)
        self.hid_state1,_=theano.scan(fn=self.computation,sequences=[self.sent1],outputs_info=[T.zeros_like(self.b_inp_hid)])
        self.hid1=self.hid_state1[-1]
        self.hid_state2,_=theano.scan(fn=self.computation,sequences=[self.sent2],outputs_info=[T.zeros_like(self.b_inp_hid)])
        self.hid2=self.hid_state2[-1]
        self.predict=theano.function([self.sent1],self.hid1)
        self.generate_function()
        self.get_similarity = theano.function([self.sent1,self.sent2],[self.score])
        self.train = theano.function([self.sent1,self.sent2,self.similarity_score],[],updates=self.updates)


class SentEmbd_syntactic(SentEmbd):
    def __init__(self,word_vector_size,dim,dep_tags_size,visualise=False):
        super().__init__(word_vector_size,dim,visualise)
        self.W_dep=nn_utils.normal_param(std=0.1, shape=(self.dim, dep_tags_size))
        depTags1 = T.lvector('dep_tags1')
        depTags2=T.lvector('dep_tags2')
        self.hid_state1,_=theano.scan(fn=self.computation_syntactic,sequences=[self.sent1,depTags1],outputs_info=[T.zeros_like(self.b_inp_hid)])
        self.hid1=self.hid_state1[-1]
        self.hid_state2,_=theano.scan(fn=self.computation_syntactic,sequences=[self.sent2,depTags2],outputs_info=[T.zeros_like(self.b_inp_hid)])
        self.hid2=self.hid_state2[-1]
        self.params.append(self.W_dep)
        self.predict=theano.function([self.sent1,depTags1],self.hid1)
        self.generate_function()
        self.get_similarity = theano.function([self.sent1,self.sent2,depTags1,depTags2],[self.score])
        self.train = theano.function([self.sent1,self.sent2,self.similarity_score,depTags1,depTags2],[],updates=self.updates)
        self.dep_tags=utils.load_dep_tags()


    def computation_syntactic(self,wVec,dep_val,prev_hid_state):
        xt_W_dep=self.W_dep[:,dep_val]
        zt=T.nnet.sigmoid((T.dot(self.W_inp_upd_in,wVec)*xt_W_dep)+ T.dot(self.U_inp_upd_hid,prev_hid_state) + self.b_inp_upd)
        rt=T.nnet.sigmoid((T.dot(self.W_inp_res_in,wVec)*xt_W_dep)+ T.dot(self.U_inp_res_hid,prev_hid_state) + self.b_inp_res)
        curr_hid_state_int=T.tanh((T.dot(self.W_inp_hid_in,wVec)*xt_W_dep) + (rt * (T.dot(self.U_inp_hid_hid,prev_hid_state))) + self.b_inp_hid) # intermediate hidden state
        t=(zt * prev_hid_state)+((1-zt) * curr_hid_state_int) #Hidden state(ht) at timestamp t
        return t

    def trainx(self,training_dataset,sim_dataset,relatedness_scores,deptags_dataset,depTags_sim_dataset,epochs):
        for val in range(epochs):
            for num in np.arange(len(training_dataset)):
                self.train(np.array(training_dataset[num]),np.array(sim_dataset[num]),relatedness_scores[num],deptags_dataset[num],depTags_sim_dataset[num])

    def predictx(self,inp_sent,glove,dep_tags,nlp):
        vectorized_sent1,dep_tags_1=utils.get_sent_details(inp_sent.strip(),glove,dep_tags,nlp,self.word_vector_size)
        sentVector=self.predict(vectorized_sent1,dep_tags_1)
        return sentVector