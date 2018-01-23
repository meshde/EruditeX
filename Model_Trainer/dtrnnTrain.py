from Helpers import utils
from Helpers import nn_utils
import spacy
import sys
import os
import pickle
import sys
import datetime
import time

class DT_RNN_Train(object):
	def __init__(self):
		SentEmbd_type="DT_RNN_"

		n=int(sys.argv[1])
		epochs=int(sys.argv[2])
		hid_dim=int(sys.argv[3])
		
		training_dataset1,training_dataset2,relatedness_scores= self.process_input_datast()
		
		from Models import dt_rnn
		self.sent_embd=dt_rnn.Dt_RNN()
		self.params=self.sent_embd.params

		print("Building loss layer")

		self.hid1=T.vector('hid1')
		self.hid2=T.vector('hid2')
		self.similarity_score = T.dscalar('score')

		self.score = (((nn_utils.cosine_similarity(self.hid1,self.hid2) + 1)/2) * 4) + 1
		self.loss = T.sqrt(abs(T.square(self.score)-T.square(self.similarity_score)))
		self.updates = lasagne.updates.adadelta(self.loss, self.params) #BlackBox

		self.train = theano.function([self.hid1,self.hid2,self.similarity_score],[],updates=self.updates)

		self.get_similarity = theano.function([self.hid1,self.hid2],[self.score])

		sent_tree_set1=[]
		sent_tree_set2=[]
		nlp = spacy.load('en')
		for num in range(len(training_dataset1)):
			sent_tree1=utils.get_dtree(training_dataset1[num].strp())
			sent_tree2=utils.get_dtree(training_dataset2[num].strp())

			sent_tree1_inputs=sent_tree1.get_rnn_input()
			sent_tree2_inputs=sent_tree2.get_rnn_input()

			sent_tree_set1.append(sent_tree1_inputs)
			sent_tree_set2.append(sent_tree2_inputs)

		for epoch_val in range(epochs):
			for num in range(len(sent_tree_set1)):
				self.training(sent_tree_set1[num],sent_tree_set2[num],relatedness_scores[num],epoch_val)

			z=str(datetime.datetime.now()).split(' ')
			file_name = SentEmbd_type+str(epoch_val+1)+"_"+str(n)+"_"+str(hid_dim)+"_"+z[0]+"_"+z[1].split('.')[0]+".txt"
			logs_path = os.path.join(os.path.join(os.path.join(BASE,'logs'),'SentEmbd'),file_name)

			print("Testing")

			acc=self.testing(sent_tree_set1[n+1:],sent_tree_set2[n+1:],relatedness_scores[n+1:],logs_path)
			acc="{0:.3}".format(acc)
			acc+="%"
			
			file_name=SentEmbd_type+str(val+1)+"_"+str(n)+"_"+str(hid_dim)+"_"+acc+"_"+z[0]+"_"+z[1].split('.')[0]+".pkl"
			save_path = os.path.join(os.path.join(os.path.join(BASE,'states'),'SentEmbd'),file_name)       
				
			self.sent_embd.save_params(save_path,epochs)



	def process_input_datast(self):

		BASE = os.path.dirname(os.path.abspath(__file__))

		dataset=[]
		training_dataset1=[]
		training_dataset2=[]
		relatedness_scores=[]

		training_set=os.path.join(os.path.join(BASE,'data'),"SICK.txt")
		with open(training_set,'r') as file1:
			raw_dataset=file1.read().split('\n')
		file1.close()

		raw_dataset=raw_dataset[1:-1]

		for item in raw_dataset:
			if item is "":
				continue
			temp=item.split('\t')
			temp2=temp[4]
			temp=temp[1:3]
			temp.append(temp2.strip())
			dataset.append(temp)


		for item in dataset:
			# print(item[0])
			# print(item[1])
			# print(item[2])
			training_dataset1.append(utils.get_dtree(item[0].strip()))
			training_dataset2.append(utils.get_dtree(item[1].strip()))
			relatedness_scores.append(float(item[2]))


		return training_dataset1,training_dataset2,relatedness_scores

	def testing(self,sent_tree1_inputs,sent_tree2_inputs,relatedness_scores,log_file):
		avg_acc=0.0
		with open(log_file,'w') as f:
			for num in np.arange(len(sent_tree_set1)):

				sent_embedding1=self.sent_embd.get_sentence_embedding(sent_tree1_inputs)
				sent_embedding2=self.sent_embd.get_sentence_embedding(sent_tree2_inputs)

				score = self.get_similarity(sent_embedding1,sent_embedding2,relatedness_scores[num])
				f.write("Actual Similarity: "+str(score)+"\n")
				f.write("Expected Similarity(From SICK.txt): "+str(relatedness_scores[num])+"\n")
				avg_acc += (abs(score[0]-relatedness_scores[num])/relatedness_scores[num])
			avg_acc =(avg_acc/len(training_dataset) * 100)
			f.write("Average Accuracy: "+str(avg_acc)+"\n")
			return avg_acc

	def training(self,sent_tree_set1,sent_tree_set2,score,epoch_val):
		print("Training")
		start = time.time()
		for num in range(n):
			sent_embedding1=self.sent_embd.get_sentence_embedding(sent_tree_set1[num])
			sent_embedding2=self.sent_embd.get_sentence_embedding(sent_tree_set2[num])

			self.train(sent_embedding1,sent_embedding2,scores)

		print("Completed Epoch %d"%(epoch_val+1))
		print("Time taken for training:\t"+str(time.time()-start))
		return


def main():
	SentEmbdTrainer=DT_RNN_Train()


if __name__ == '__main__':
	main()

















