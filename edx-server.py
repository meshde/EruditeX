import subprocess
import os
import nltk.data

from flask import Flask
from flask import Response
from flask import request

from Helpers import file_extraction as filer
from Helpers import deployment_utils as deploy 
from IR import infoRX
from Models import abcnn_model
from Models import AnsSelect
from Models import DT_RNN

class EdXServer():

	def __init__(self):
		self.file = ''
		self.context = []
		self.query = '';	

	def get_file(self, filename):

		# print(filename)
		self.file = os.path.join('.\data-og\corpus\\'+filename)

		self.context = filer.extract_file_contents(self.file)
		if len(self.context) > 0:
			return True

		return True	#TODO: remove before deploy

	def get_query(self, query):
		
		self.query = query
		print(self.query)

		# Filter top 5 paras using Info Retrieval
		para_select = infoRX.retrieve_info(self.context, self.query)
		para_sents = []
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		for para in para_select:
			para_sents.extend(tokenizer.tokenize(para))

		# Select Ans Sents - ABCNN
		ans_sents = abcnn.ans_select(query, para_sents)

		# TODO: Phase 2-3: Input Module and Answer Module
		for ans in ans_sents:
			words = deploy.extract_answer_from_sentence(ans, self.query)
			

		# proc = subprocess.Popen(['python','test.py',query],shell=False,stdout=subprocess.PIPE)
		answer = ''
		return answer

app = Flask(__name__)
server = EdXServer()
abcnn = abcnn_model()
ans_select = AnsSelect()
dt_rnn = DT_RNN()

@app.route('/filed',methods=['POST'])
def filer():
	filename = request.get_json(force=True)['filename']
	if server.get_file(filename):
		resp = Response('Context Ready.')
		resp.headers['Access-Control-Allow-Origin'] = '*'
		return resp


@app.route('/query',methods=['POST'])
def queried():
	query = request.get_json(force=True)['query']
	resp = Response(server.get_query(query))
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp

if __name__ == '__main__':
	app.run()
