import sys
sys.path.append("../")
from Helpers import utils
from Helpers import nn_utils
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

class DT_RNN_Train(object):
	def __init__(self,load_input=""):
		SentEmbd_type="DT_RNN_"

		self.n=int(sys.argv[1])
		self.epochs=int(sys.argv[2])
		self.hid_dim=int(sys.argv[3])

		print("Pre-Processing Data Set:")

		if(load_input==""):
			training_dataset1,training_dataset2,relatedness_scores= self.process_input_datast()
		else:
			training_dataset1=pickle.load(open('training_set1.p','rb'))
			training_dataset2=pickle.load(open('training_set2.p','rb'))
			relatedness_scores=pickle.load(open('scores.p','rb'))
		
		from Models import dt_rnn
		self.sent_embd=dt_rnn.DT_RNN()
		self.params=self.sent_embd.params

		inputs1=self.sent_embd.get_graph_input()
		inputs2=self.sent_embd.get_graph_input()

		assert(inputs1!=inputs2)
		test_inputs=self.sent_embd.get_graph_input()

		

		print("Building loss layer")

		sentence_embedding1, self.hid1=self.sent_embd.get_theano_graph(inputs1)
		sentence_embedding2, self.hid2=self.sent_embd.get_theano_graph(inputs2)

		sentence_embedding3, self.hid3=self.sent_embd.get_theano_graph(test_inputs)
		self.get_sentence_embedding = theano.function(test_inputs, sentence_embedding3)
		# self.get_hidden_states = theano.function(inputs, hidden_states)

		self.similarity_score = T.dscalar('score')


		self.score = (((nn_utils.cosine_similarity(self.hid1,self.hid2) + 1)/2) * 4) + 1
		self.loss = T.sqrt(abs(T.square(self.score)-T.square(self.similarity_score)))
		self.updates = lasagne.updates.adadelta(self.loss, self.params) #BlackBox

		inputs=[]

		inputs.extend(inputs1)
		inputs.extend(inputs2)
		inputs.append(self.similarity_score)

		print(inputs)

		self.train = theano.function(inputs,[],updates=self.updates)

		self.get_similarity = theano.function([self.hid1,self.hid2],[self.score])

		sent_tree_set1=[]
		sent_tree_set2=[]

		if(load_input==""):
			for num in range(len(training_dataset1)):
				sent_tree1=training_dataset1[num]
				sent_tree2=training_dataset2[num]

				sent_tree1_inputs=sent_tree1.get_rnn_input()
				sent_tree2_inputs=sent_tree2.get_rnn_input()


				sent_tree_set1.append(sent_tree1_inputs)
				sent_tree_set2.append(sent_tree2_inputs)

			pickle.dump(sent_tree_set1, open( "sent_tree_set1.p", "wb" ) )
			pickle.dump(sent_tree_set2, open( "sent_tree_set2.p", "wb" ) )

		else:
			sent_tree_set1=pickle.load(open('sent_tree_set1.p','rb'))
			sent_tree_set2=pickle.load(open('sent_tree_set2.p','rb'))



		# print(sent_tree_set1[0])

		for epoch_val in range(self.epochs):
			self.training(sent_tree_set1[:self.n],sent_tree_set2[:self.n],relatedness_scores[:self.n],epoch_val)

			z=str(datetime.datetime.now()).split(' ')
			file_name = SentEmbd_type+str(epoch_val+1)+"_"+str(self.n)+"_"+str(self.hid_dim)+"_"+z[0]+"_"+z[1].split('.')[0]+".txt"
			logs_path = os.path.join(os.path.join(os.path.join(BASE,'logs'),'SentEmbd'),file_name)

			print("Testing")

			acc=self.testing(sent_tree_set1[self.n+1:],sent_tree_set2[self.n+1:],relatedness_scores[self.n+1:],logs_path)
			acc="{0:.3}".format(acc)
			acc+="%"
			
			file_name=SentEmbd_type+str(val+1)+"_"+str(self.n)+"_"+str(self.hid_dim)+"_"+acc+"_"+z[0]+"_"+z[1].split('.')[0]+".pkl"
			save_path = os.path.join(os.path.join(os.path.join(BASE,'states'),'SentEmbd'),file_name)       
				
			self.sent_embd.save_params(save_path,self.epochs)



	def process_input_datast(self):

		nlp = spacy.load('en')

		BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

		dataset=[]
		training_dataset1=[]
		training_dataset2=[]
		relatedness_scores=[]

		training_set=os.path.join(os.path.join(BASE,'data'),"SICK.txt")
		with open(training_set,'r') as file1:
			raw_dataset=file1.read().split('\n')
		file1.close()

		raw_dataset=raw_dataset[1:-1]

		val=0

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
			training_dataset1.append(utils.get_dtree(item[0].strip(),nlp))
			training_dataset2.append(utils.get_dtree(item[1].strip(),nlp))
			relatedness_scores.append(float(item[2]))

			print("Pre-processed pair %d"%(val+1))
			val+=1

		pickle.dump(training_dataset1, open( "training_set1.p", "wb" ) )
		pickle.dump(training_dataset2, open( "training_set2.p", "wb" ) )
		pickle.dump(relatedness_scores, open( "scores.p", "wb" ) )	
		


		return training_dataset1,training_dataset2,relatedness_scores

	def testing(self,sent_tree1_inputs,sent_tree2_inputs,relatedness_scores,log_file):
		avg_acc=0.0
		with open(log_file,'w') as f:
			for num in np.arange(len(sent_tree_set1)):

				sent_embedding1=self.get_sentence_embedding(sent_tree1_inputs[num])
				sent_embedding2=self.get_sentence_embedding(sent_tree2_inputs[num])

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
		for num in range(self.n):
			inputs1= sent_tree_set1[num]
			inputs2=sent_tree_set2[num]

			# print(inputs1[0])
			# print(inputs1[1])
			# print(inputs1[2])
			# print(inputs1[3])

			# print(inputs2[0])
			# print(inputs2[1])
			# print(inputs2[2])
			# print(inputs2[3])

			self.train(inputs1[0], inputs1[1], inputs1[2], inputs1[3], inputs2[0], inputs2[1], inputs2[2], inputs2[3], score[num])


		print("Completed Epoch %d"%(epoch_val+1))
		print("Time taken for training:\t"+str(time.time()-start))
		return


def main():
	SentEmbdTrainer=DT_RNN_Train("load fom file")


if __name__ == '__main__':
	main()

















