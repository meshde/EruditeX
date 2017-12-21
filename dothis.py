import sys
from Helpers import utils
import os
from Models import dmn_basic
import time


def babi1():
	queries = ['where is john','where is daniel','where is sandra','where is mary']
	glove = utils.load_glove()
	for file in os.listdir('states/dmn_basic/'):
		if 'babi1' in file:
			print(file)
			for query in queries:
				quest = utils.init_babi_deploy(os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'data'),'corpus'),'babi.txt'),query)
				dmn = dmn_basic.DMN_basic(babi_train_raw=quest,babi_test_raw=[],word2vec=glove,word_vector_size=50,dim=40,mode='deploy',answer_module='feedforward', input_mask_mode="sentence", memory_hops=5, l2=0, 
			                normalize_attention=False, answer_vec='index', debug=False)
				dmn.load_state(os.path.join('states/dmn_basic/',file))

				prediction = dmn.step_deploy()

				prediction = prediction[0][0]
				print(query)
				for ind in prediction.argsort()[::-1]:
					if ind < dmn.answer_size:
						print(dmn.ivocab[ind])
						break
def babi2():
	# start = time.time()
	queries = ['who is in the office','where is the milk','where is sandra','where is mary']
	glove = utils.load_glove()
	for file in os.listdir('states/dmn_basic/'):
		if 'babi2' in file:
			print(file)
			for query in queries:
				quest = utils.init_babi_deploy(os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'data'),'corpus'),'babi3.txt'),query)
				dmn = dmn_basic.DMN_basic(babi_train_raw=quest,babi_test_raw=[],word2vec=glove,word_vector_size=50,dim=40,mode='deploy',answer_module='feedforward', input_mask_mode="sentence", memory_hops=5, l2=0, 
			                normalize_attention=False, answer_vec='index', debug=False)
				dmn.load_state(os.path.join('states/dmn_basic/',file))

				prediction = dmn.step_deploy()

				prediction = prediction[0][0]
				print(query)
				for ind in prediction.argsort()[::-1]:
					if ind < dmn.answer_size:
						print(dmn.ivocab[ind])
						break
	# print('Time taken:',time.time()-start)
	# print(len(dmn.ivocab))
	# print(len(dmn.vocab))
	# print(dmn.answer_size)
	# print(prediction.argmax())
	# print(len(prediction[0][0]))
	# print(prediction.shape)
if __name__ == '__main__':
	babi1()