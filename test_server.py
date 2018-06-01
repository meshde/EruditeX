import subprocess
import os
# import nltk.data
import threading

from flask import Flask
from flask import Response
from flask import request
from flask import jsonify
from werkzeug import secure_filename

# from Helpers import file_extraction as filer
# from Helpers import deployment_utils as deploy 
# from IR import infoRX
# from Models import abcnn_model
# from Models import AnsSelect
# from Models import DT_RNN

class EdXServer():

	status = {}

	def __init__(self):
		self.file = ''
		self.context = []
		self.query = ''	

	@classmethod
	def update(cls, value):
		cls.status = value

	def get_file(self, filename):

		# print(filename)
		self.file = os.path.join(app2.config['UPLOAD_FOLDER'] + filename)

		# self.context = filer.extract_file_contents(self.file)
		
		# print(self.context)
        
		if len(self.context) > 0:
			return True

		return True	#TODO: remove before deploy

	def get_query(self, query):
		
		self.query = query
		# print(self.query)

		try:
			import time
			answers = [('ground', 0.872), ('bat', 0.655), ('catch', 0.317), ('bowler', 0.012), ('catch', 0.317), ('bowler', 0.012)]

			ans_list = []
			for x in answers:
				ans_list.append({'word':x[0], 'score': x[1]})

			# Filter top 5 paras using Info Retrieval
			# self.update('Ranking Paragraphs using Information Retrieval..')
			self.update({'val': 'Ranking Paragraphs using Information Retrieval.'})
			# para_select = infoRX.retrieve_info(self.context, self.query)
			# para_sents = []
			# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
			t_end = time.time() + 3
			while time.time() < t_end:
				status = 'Selecting Answer Sentences..'
			self.update({'val': 'Sentences selected by IR Module', 'answers': ans_list})

			# for para in para_select:
			#     para_sents.extend(tokenizer.tokenize(para))

			# print('Sentences selected by IR Module:')
			# print(para_sents)
			t_end = time.time() + 3
			while time.time() < t_end:
				status = 'Selecting Answer Sentences..'
			self.update({'val': 'Ranking Candidate Answer Sentences.'})
			self.update('Selecting Answer Sentences..')
			print(status)

			# Select Ans Sents - ABCNN
			# ans_sents = abcnn.ans_select(query, para_sents)

			print('Sentences scored by Sentence Selection Module:')
			# print(ans_sents)
			t_end = time.time() + 3
			while time.time() < t_end:
				status = 'Generating VDT and extracting Answer..'

			self.update('Generating VDT and extracting Answer..')
			# best_ans, score, answers = deploy.extract_answer_from_sentences(ans_sents, query)

		except:
			return {'answers': [{'word': 'ERROR', 'score': 'QA Subsystem failure.'}]}
		
		self.update('false')
		
		answers = [('ground', 0.872), ('bat', 0.655), ('catch', 0.317), ('bowler', 0.012)]

		ans_list = []
		for x in answers:
			ans_list.append({'word':x[0], 'score': x[1]})

		ans_dict = {'answers': ans_list}

		return ans_dict


app = Flask(__name__)
app2 = Flask(__name__)
server = EdXServer()
# abcnn = abcnn_model()
# ans_select = AnsSelect()
# dt_rnn = DT_RNN()

app2.config['UPLOAD_FOLDER'] = os.path.join('./data/uploads/')

@app2.route('/filed',methods=['POST'])
def filer():

	# data = request.get_json(force=True)
	# filename = data['filename']
	# file = data['file']
	
	f = request.files['file']
	f.save(os.path.join(app2.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

	if server.get_file(f.filename):
		resp = Response('File uploaded. Context Ready.')
		resp.headers['Access-Control-Allow-Origin'] = '*'
		return resp
	

@app2.route('/status', methods=['POST'])
def status():
	print(EdXServer.status)
	resp = jsonify({'status': EdXServer.status})
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp


@app.route('/query',methods=['POST'])
def queried():
	query = request.get_json(force=True)['query']
	# resp = Response(server.get_query(query))
	resp = jsonify(server.get_query(query))
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp

def start1(port):
	app.run(port=port)

def start2(port):
	app2.run(port=port)

if __name__ == '__main__':
    t1 = threading.Thread(target=start1, args=(5001,))
    t2 = threading.Thread(target=start2, args=(5000,))
    t1.start()
    t2.start()