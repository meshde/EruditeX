#Sentence Embedder.
import utils
import nn_utils
import numpy as np
import theano.tensor as T
import theano
import lasagne
import os
import pickle
def load_glove(dim = 50): #Redundant
    glove = {}
    # path = "/Users/meshde/Mehmood/EruditeX/data/glove/glove.6B.50d.txt"
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/glove/glove.6B.%sd.txt' % dim)
    with open(path,'r') as f:
        for line in f:
            l = line.split()
            glove[l[0]] = list(map(float,l[1:]))
    return glove


def get_vector(word,glove): #Redundant
    try:
        ans = np.array(glove[word]).reshape((1,50))
        return ans
    except:
        return np.random.rand(1,50)

def constant_param(value=0.0, shape=(0,)): #Redundant
    return theano.shared(lasagne.init.Constant(value).sample(shape), borrow=True)

def normal_param(std=0.1, mean=0.0, shape=(0,)): #Redundant
    return theano.shared(lasagne.init.Normal(std, mean).sample(shape), borrow=True)

def cosine_similarity(A,B): #Redundant
	return T.dot(A,T.transpose(B))/(T.dot(A,T.transpose(A))*T.dot(B,T.transpose(B)))

class SentEmbd(object):
    def __init__(self,word_vector_size,dataset_size,dim=50):
        self.dim=dim #Dimmensions of Hidden State of the GRU
        self.W_inp_res_in = normal_param(std=0.1, shape=(self.dim, word_vector_size))
        self.U_inp_res_hid = normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_upd_in = normal_param(std=0.1, shape=(self.dim, word_vector_size))
        self.U_inp_upd_hid = normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_hid_in = normal_param(std=0.1, shape=(self.dim, word_vector_size))
        self.U_inp_hid_hid = normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid = constant_param(value=0.0, shape=(self.dim,))

        self.hid_state_matrix=np.zeros(shape=(dataset_size+1,dim))
        self.hid_state_matrix_exp=np.zeros(shape=(dataset_size+1,dim))
        self.count=0 # For keeping track of which training pair is being used in the current epoch.
        self.dummy_hid_state=T.zeros(np.zeros((1,50)).shape,dtype=theano.config.floatX)



        # Creating A GRU using theano
        wordvec=T.dvector('xt')
        prev_hid_state=T.dvector('ht-1')
        temp=T.dvector('ht')
        similarity_score = T.dscalar('score')
        sent1=T.dmatrix('sent1')
        sent2=T.dmatrix('sent2')
        hid_state1,_=theano.scan(fn=self.computation,sequences=[sent1],outputs_info=[T.zeros_like(self.b_inp_hid)])
        self.hid1=hid_state1[-1]
        hid_state2,_=theano.scan(fn=self.computation,sequences=[sent2],outputs_info=[T.zeros_like(self.b_inp_hid)])
        self.hid2=hid_state2[-1]
        # print(type(self.hid1))
        score = cosine_similarity(self.hid1,self.hid2)
        # print(score.shape.eval({sent1:np.ones((10,50)),sent2: np.ones((10,50))}))
        self.loss = T.sqrt(T.square(score)-T.square(similarity_score))

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
        updates = lasagne.updates.adadelta(self.loss, self.params) #BlackBox

        self.train = theano.function([sent1,sent2,similarity_score],[],updates=updates)
        self.predict = theano.function([sent1],[hid_state1])
    def computation(self,wVec,prev_hid_state):
        zt=T.nnet.sigmoid(T.dot(self.W_inp_upd_in,wVec)+ T.dot(self.U_inp_upd_hid,prev_hid_state) + self.b_inp_upd)
        rt=T.nnet.sigmoid(T.dot(self.W_inp_res_in,wVec)+ T.dot(self.U_inp_res_hid,prev_hid_state) + self.b_inp_res)
        curr_hid_state_int=T.tanh(T.dot(self.W_inp_hid_in,wVec) + (rt * (T.dot(self.U_inp_hid_hid,prev_hid_state))) + self.b_inp_hid) # intermediate hidden state
        t=(zt * prev_hid_state)+((1-zt) * curr_hid_state_int) #Hidden state(ht) at timestamp t
        return t

    def trainx(self,training_dataset,exp_dataset,relatedness_scores):
        for num in np.arange(len(training_dataset)):
            self.train(np.array(training_dataset[num]).reshape((-1,50)),np.array(exp_dataset[num]).reshape((-1,50)),relatedness_scores[num])
            # print("Trained on Sentence Pair ",(num+1))

    def predictx(self,inp_sent):
        # print(np.array(inp_sent).reshape((-1,50)).shape)
        hidden_states=self.predict(np.array(inp_sent).reshape((-1,50)))
        print hidden_states
    def testing(self,sent1,sent2,exp_sccore):
        hid1=self.predict(np.array(sent1).reshape((-1,50)))
        hid2=self.predict(np.array(sent2).reshape((-1,50)))
        score=cosine_similarity(hid1,hid2)
        print("Actual Similarity: ",score)
        print("Expected Similarity: ",exp_sccore)

    def printParams(self):
        print self.W_inp_upd_in.get_value()





#PreProcessing of Data before training our SentEmbd Model includes converting of words to their vector representation
training_set="/home/mit/Desktop/EruditeX/SICK.txt"
file = open(training_set,'r')
raw_dataset=file.read().split('\n')
n=input("Enter the no. training examples to learn from: ")
# print(raw_dataset) #TESTING PURPOSE
dataset=[]
training_dataset=[]
sim_dataset=[]
relatedness_scores=[]
raw_dataset=raw_dataset[1:100]
for item in raw_dataset:
    temp=item.split('\t')
    temp2=temp[4]
    # print(temp2)
    temp=temp[1:3]
    temp.append(temp2)
    dataset.append(temp)
    # print(temp)
# print(dataset)
glove=load_glove()
# print("Word vector for And:")
# print(get_vector('and',glove))
for item in dataset:
    sent1=item[0].split(' ')
    sent2=item[1].split(' ')
    sent_1=[]
    sent_2=[]
    for word in sent1:
        sent_1.append(get_vector(word,glove))
    for word in sent2:
       sent_2.append(get_vector(word,glove))
    training_dataset.append(sent_1)
    sim_dataset.append(sent_2)
    relatedness_scores.append(float(item[2]))



sent_embd=SentEmbd(50,len(dataset)) #GRU INITIALIZED
# batch_size=input("Enter the batch size in which the training dataset has to be divided: ")
# epochs=input("Enter the number of epochs needed to train the GRU: ")
batch_size=1
epochs=1
# print(np.array(training_dataset[0]).reshape((-1,50)).shape)
# print(np.array(relatedness_scores))
print "Before Training:"
sent_embd.printParams()
sent_embd.trainx(training_dataset[:n],sim_dataset[:n],relatedness_scores[:n]) #Training THE GRU using the SICK dataset
print "After Training:"
sent_embd.printParams()
sent_embd.predictx(training_dataset[n+1])
sent_embd.testing(training_dataset[n+1],sim_dataset[n+1],relatedness_scores[n+1])

# # Saving the trained Model:
# pickle.dump( sent_embd, open( "pre_trained_model", "wb" ) )
