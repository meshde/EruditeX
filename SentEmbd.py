#Sentence Embedder.
import utils
import nn_utils
import numpy as np
import theano.tensor as T
import theano
import lasagne
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


class SentEmbd:
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
        self.hid_state_matrix_out=np.zeros(shape=(dataset_size+1,dim))
        self.count=0 # For keeping track of which training pair is being used in the current epoch.


        # Creating A GRU using theano
        wordvec=T.dvector('xt')
        prev_hid_state=T.vector('ht-1')
        temp=T.vector('ht')
        zt=T.nnet.sigmoid(T.dot(self.W_inp_upd_in,wordvec)+ T.dot(self.U_inp_upd_hid,prev_hid_state) + self.b_inp_upd)
        rt=T.nnet.sigmoid(T.dot(self.W_inp_res_in,wordvec)+ T.dot(self.U_inp_res_hid,prev_hid_state) + self.b_inp_res)
        curr_hid_state_int=T.tanh(T.dot(self.W_inp_hid_in,wordvec) + np.multiply(T.dot(self.U_inp_hid_hid,prev_hid_state),rt) +self.b_inp_hid) # intermediate hidden state
        #TODO
        t=np.multiply(zt,prev_hid_state)+np.multiply((1-zt),curr_hid_state_int) #Hidden state(ht) at timestamp t

        self.curr_hid_state = theano.function([wordvec,prev_hid_state],t)

    def computation(self,wVec_in,wVev_out):
        self.count += 1
        self.hid_state_matrix[self.count] = self.curr_hid_state(wVec_in,self.hid_state_matrix[self.count-1])
        self.hid_state_matrix_out[self.count]=self.curr_hid_state(wVec_out,self.hid_state_matrix[self.count-1])
        return 1.0

    def train_batch(self,batch_in,batch_out):
        theano.scan(fn=self.computation,sequences=[batch_in,batch_out],outputs_info=T.dscalar())
        cos_sim=cosine_similarity(self.hid_state_matrix[count],self.hid_state_matrix_out[self.count]) #A measure of similarity between two sentences passed through our model.
        #TODO exp_similarity=? #Obtain value from sick dataset
        #TODO Calculate loss and send it back to function train() where weights will be trained.


    def train(self,training_dataset,output_dataset,batch_size,epochs):
        start=0
        end=0
        for val in range(epochs):
            self.count=0
            batch_in=[]
            batch_out=[]
            while(True):
                if(start>=len(dataset)):
                    break
                end=start+batch_size-1
                if(end>=len(dataset)):
                    batch_in=training_dataset[start:]
                    batch_out=output_dataset[start:]
                else:
                    batch_in=training_dataset[start:end]
                    batch_out=output_dataset[start:end]
                start=end+1

                #Training GRU on each batch
                self.train_batch(batch_in,batch_out)

            print("Completed ",val+1," epoch/s")



#PreProcessing of Data before training our SentEmbd Model includes converting of words to their vector representation
training_set="/home/mit/Desktop/EruditeX/SICK.txt"
file = open(training_set,'r')
raw_dataset=file.read().split('\n')
raw_dataset=raw_dataset[1:20]
print(raw_dataset) #TESTING PURPOSE
dataset=[]
training_dataset=[]
output_dataset=[]
for item in raw_dataset:
    temp=item.split('\t')
    temp=temp[1:3]
    dataset.append(temp)
# print(dataset)
glove=load_glove
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
    output_dataset.append(sent_2)



sent_embd=SentEmbd(50,len(dataset)) #GRU INITIALIZED
batch_size=input("Enter the number of batches in which the training dataset has to be divided: ")
epochs=input("Enter the number of epochs needed to train the GRU: ")
sent_embd.train(training_dataset,output_dataset,batch_size,epochs) #Training THE GRU using the SICK dataset
