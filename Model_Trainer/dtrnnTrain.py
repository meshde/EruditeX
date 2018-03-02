import sys
# sys.path.append("../")
from Helpers import utils
from Helpers import nn_utils
from Helpers import path_utils
import spacy
import sys
import os
import pickle
import theano
import theano.tensor as T
import datetime
import time
import pickle
import lasagne
import numpy as np
from sklearn.utils import shuffle

class DT_RNN_Train(object):

	def __init__(self, n=None, epochs=None, hid_dim=None, optimizer='adadelta'):
		self.SentEmbd_type="DT_RNN_"

		if not n:
			self.n=int(sys.argv[1])
		else:
			self.n = n
		if not epochs:
			self.epochs=int(sys.argv[2])
		else:
			self.epochs = epochs
		if not hid_dim:
			self.hid_dim=int(sys.argv[3])
		else:
			self.hid_dim = hid_dim

		optimization_function = nn_utils.get_optimization_function(optimizer)

		from Models import dt_rnn
		self.sent_embd = dt_rnn.DT_RNN(dim=self.hid_dim)
		self.params = self.sent_embd.params

		inputs1 = self.sent_embd.get_graph_input()
		inputs2 = self.sent_embd.get_graph_input()

		assert(inputs1!=inputs2)

		
		print("Building loss layer")
		sentence_embedding1, self.hid1=self.sent_embd.get_theano_graph(inputs1)
		sentence_embedding2, self.hid2=self.sent_embd.get_theano_graph(inputs2)

		self.similarity_score = T.dscalar('score')


		self.score = (((nn_utils.cosine_similarity(self.hid1,self.hid2) + 1)/2) * 4) + 1
		self.loss = T.sqrt(abs(T.square(self.score)-T.square(self.similarity_score)))
		self.grad = theano.grad(self.loss, self.params)
		self.updates = optimization_function(self.loss, self.params) #BlackBox
		
		inputs=[]
		inputs.extend(inputs1)
		inputs.extend(inputs2)
		inputs.append(self.similarity_score)
		# print(inputs)

		self.train = theano.function(inputs, [self.loss],updates=self.updates)
		self.get_similarity = theano.function(inputs,[self.score],on_unused_input='ignore')
		self.get_grad = theano.function(inputs, self.grad)



	def train_dtRNN(self):
		print("Loading Pre-processed SICK dataset")

		sick_path = path_utils.get_sick_path()
		self.load_dataset(sick_path)

		print("Load Complete")

		BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

		for epoch_val in range(self.epochs):
			sent_tree_set1, sent_tree_set2, relatedness_scores, sick_text = shuffle(self.sent_tree_set1, self.sent_tree_set2, self.relatedness_scores, self.sick_text)
			self.training(sent_tree_set1[:self.n], sent_tree_set2[:self.n], relatedness_scores[:self.n], epoch_val)

			z=str(datetime.datetime.now()).split(' ')
			file_name = self.SentEmbd_type+str(epoch_val+1)+"_"+str(self.n)+"_"+str(self.hid_dim)+"_"+z[0]+"_"+z[1].split('.')[0]+".txt"
			logs_path = path_utils.get_logs_path('SentEmbd/'+file_name)


			print("Testing")

			acc = self.testing(self.sent_tree_set1[self.n:],self.sent_tree_set2[self.n:],self.relatedness_scores[self.n:],logs_path)
			acc = "{0:.3}".format(acc)
			acc += "%"

			print("Accuracy after epoch %d is %s"%(epoch_val+1,acc))
			
			file_name = self.SentEmbd_type+str(epoch_val+1)+"_"+str(self.n)+"_"+str(self.hid_dim)+"_"+acc+"_"+z[0]+"_"+z[1].split('.')[0]+".pkl"
			save_path = path_utils.get_save_states_path('SentEmbd/'+file_name)       
				
			self.sent_embd.save_params(save_path,self.epochs)
		return

	def toposort(self):
		ans =self.get_grad.maker.fgraph.toposort()
		return ans

	def testing(self, sent_tree_set1, sent_tree_set2, relatedness_scores, log_file):
		avg_acc=0.0
		with open(log_file,'w') as f:
			inputs=[]
			for num in np.arange(len(sent_tree_set1)):
				sent1 = sent_tree_set1[num]
				sent2 = sent_tree_set2[num]

				score = self.get_similarity(np.array(sent1['word_vectors']), np.array(sent1['parent_indices']), np.array(sent1['is_leaf']), np.array(sent1['dep_tags']), np.array(sent2['word_vectors']), np.array(sent2['parent_indices']), np.array(sent2['is_leaf']), np.array(sent2['dep_tags']),relatedness_scores[num])

				f.write("Actual Similarity: "+str(score)+"\n")
				f.write("Expected Similarity(From SICK.txt): "+str(relatedness_scores[num])+"\n")
				avg_acc += (abs(score[0]-relatedness_scores[num])/relatedness_scores[num])
			avg_acc =(avg_acc/len(sent_tree_set1) * 100)
			f.write("Average Accuracy: "+str(avg_acc)+"\n")
			return avg_acc

	def load_dataset(self, sick_path):

		try:
			SICK_dataset_dtree,_=pickle.load(open(sick_path,"rb"))
		except:
			print("Invalid CACHED SICK FILE PATH: {}".format(sick_path))
			sys.exit()

		sent_tree_set1=[]
		sent_tree_set2=[]
		relatedness_scores=[]
		sick_text=[]

		for entry in SICK_dataset_dtree:
			entryA=entry['A']
			entryB=entry['B']

			temp=[]

			# combined_entryA=[]
			# combined_entryA.extend([entryA['word_vectors'], entryA['parent_indices'], entryA['is_leaf'], entryA['dep_tags']])

			# combined_entryB=[]
			# combined_entryB.extend([entryB['word_vectors'], entryB['parent_indices'], entryB['is_leaf'], entryB['dep_tags']])

			sent_tree_set1.append(entryA)
			sent_tree_set2.append(entryB)
			relatedness_scores.append(entry['score'])

			sick_text.append(temp.extend([entryA['text'],entryB['text'],entry['score']]))		

		self.sent_tree_set1 = sent_tree_set1
		self.sent_tree_set2 = sent_tree_set2
		self.relatedness_scores = relatedness_scores
		self.sick_text = sick_text
		return

	def training(self,sent_tree_set1,sent_tree_set2,score,epoch_val):
		print("Training: Epoch ",epoch_val+1)
		start = time.time()
		for num in range(self.n):
			sent1 = sent_tree_set1[num]
			sent2 = sent_tree_set2[num]

			# print("Printing inputs for debugging purpose")
			# print("Input set 1:")
			# print(np.array(sent1['word_vectors']).shape)
			# print(np.array(sent1['parent_indices']).shape)
			# print(np.array(sent1['is_leaf']).shape)
			# print(np.array(sent1['dep_tags']).shape)

			# print("Printing inputs for debugging purpose")
			# print("Input set 2:")
			# print(np.array(sent2['word_vectors']).shape)
			# print(np.array(sent2['parent_indices']).shape)
			# print(np.array(sent2['is_leaf']).shape)
			# print(np.array(sent2['dep_tags']).shape)

			self.train(np.array(sent1['word_vectors']), np.array(sent1['parent_indices']), np.array(sent1['is_leaf']), np.array(sent1['dep_tags']), np.array(sent2['word_vectors']), np.array(sent2['parent_indices']), np.array(sent2['is_leaf']), np.array(sent2['dep_tags']), score[num])

			# self.get_grad(np.array(sent1['word_vectors']), np.array(sent1['parent_indices']), np.array(sent1['is_leaf']), np.array(sent1['dep_tags']), np.array(sent2['word_vectors']), np.array(sent2['parent_indices']), np.array(sent2['is_leaf']), np.array(sent2['dep_tags']), score[num])


		print("Completed Epoch %d"%(epoch_val+1))
		print("Time taken for training:\t"+str(time.time()-start))
		return


# def main():
#     SentEmbdTrainer=DT_RNN_Train("load")

# 			# for debugging purpose
# 			print("Printing input shapes")

# 			print("Input set 1:")
# 			print(np.array(inputs1[0]).shape)
# 			print(np.array(inputs1[1]).shape)
# 			print(np.array(inputs1[2]).shape)
# 			print(np.array(inputs1[3]).shape)

# 			print("Input set 2:")
# 			print(np.array(inputs2[0]).shape)
# 			print(np.array(inputs2[1]).shape)
# 			print(np.array(inputs2[2]).shape)
# 			print(np.array(inputs2[3]).shape)

# 			self.train(np.array(inputs1[0]), np.array(inputs1[1]), np.array(inputs1[2]), np.array(inputs1[3]), np.array(inputs2[0]), np.array(inputs2[1]), np.array(inputs2[2]), np.array(inputs2[3]), score[num])


# 		print("Completed Epoch %d"%(epoch_val+1))
# 		print("Time taken for training:\t"+str(time.time()-start))
# 		return


# def main():
# 	SentEmbdTrainer=DT_RNN_Train()



# if __name__ == '__main__':
#     main()

















