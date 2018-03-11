import subprocess
import os
import nltk.data

from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
from werkzeug import secure_filename

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
		self.file = os.path.join(app.config['UPLOAD_FOLDER'] + filename)

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

		best_ans, score, answers = deploy.extract_answer_from_sentences(ans_sents, query)

		# Ignore: Phase 2-3: Input Module and Answer Module
		# answers = []
		# for ans, a_score in ans_sents.iteritems():
		# 	words = deploy.extract_answer_from_sentence(ans, self.query)
		# 	words = sorted(words, key=operator.itemgetter(1))
		# 	for word, w_score in words.iteritems()[:5]:
		# 		answers.append((word, w_score * a_score))
		# answers = sorted(answers, key=operator.itemgetter(1))
		# proc = subprocess.Popen(['python','test.py',query],shell=False,stdout=subprocess.PIPE)
		
		ans_list = []
		for x in answers:
			ans_list.append({'word':x[0], 'score': x[1]})

		ans_dict = {'answers': ans_list}

		return ans_dict


app = Flask(__name__)
server = EdXServer()
abcnn = abcnn_model()
ans_select = AnsSelect()
dt_rnn = DT_RNN()

app.config['UPLOAD_FOLDER']	= os.path.join('./data/uploads')

@app.route('/filed',methods=['POST'])
def filer():

	# data = request.get_json(force=True)
	# filename = data['filename']
	# file = data['file']
	
	f = request.files['file']
	f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
	print(f)

	if server.get_file(f.filename):
		resp = Response('File uploaded. Context Ready.')
		resp.headers['Access-Control-Allow-Origin'] = '*'
		return resp
	
@app.route('/query',methods=['POST'])
def queried():
	query = request.get_json(force=True)['query']
	# resp = Response(server.get_query(query))
	resp = jsonify(server.get_query(query))
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp

if __name__ == '__main__':
	app.run()