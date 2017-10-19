import sys
import utils
import os
import dmn_basic
import time

def main():
	start = time.time()
	query = sys.argv[1]
	glove = utils.load_glove()
	quest = utils.init_babi_deploy(os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'data'),'corpus'),'babi.txt'),query)

	dmn = dmn_basic.DMN_basic(babi_train_raw=quest,babi_test_raw=[],word2vec=glove,word_vector_size=50,dim=40,mode='deploy',answer_module='feedforward', input_mask_mode="sentence", memory_hops=5, l2=0, 
                normalize_attention=False, answer_vec='index', debug=False)

	dmn.load_state('states/dmn_basic.mh5.n40.bs10.babi1.epoch2.test1.20454.state')

	prediction = dmn.step_deploy()

	prediction = prediction[0][0]
	for ind in prediction.argsort()[::-1]:
		if ind < dmn.answer_size:
			print(dmn.ivocab[ind])
			break
	print('Time taken:',time.time()-start)
	# print(len(dmn.ivocab))
	# print(len(dmn.vocab))
	# print(dmn.answer_size)
	# print(prediction.argmax())
	# print(len(prediction[0][0]))
	# print(prediction.shape)
if __name__ == '__main__':
	main()