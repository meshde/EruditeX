import subprocess
import os

from flask import Flask
from flask import Response
from flask import request

from Helpers import file_extraction as filer
from IR import infoRX

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


	def get_query(self, query):
		
		self.query = query
		print(self.query)

		para_select = infoRX.retrieve_info(self.context, self.query)

		proc = subprocess.Popen(['python','test.py',query],shell=False,stdout=subprocess.PIPE)
		
		resp = Response(proc.communicate()[0].decode())
		resp.headers['Access-Control-Allow-Origin'] = '*'
		return resp


app = Flask(__name__)
server = EdXServer()

@app.route('/filed',methods=['POST'])
def filer():
	filename = request.get_json(force=True)["filename"]
	if server.get_file(filename):
		resp = Response('Context Ready')
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
