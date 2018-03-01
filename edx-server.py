import subprocess
import os

from flask import Flask
from flask import Response
from flask import request

# from Helpers import file_extraction as filer

class EdXServer():

	def __init__(self):
		self.file = ''
		self.context = []
		self.query = '';	

	def get_file(self):
		filename = request.get_json(force=True)['filename']
		print(filename)

		self.file = os.path.join('.\data-og\corpus\\'+filename)

		# self.context = filer.extract_file_contents(self.file)	

	def get_query(self):
		
		self.query = request.get_json(force=True)['query']
		print(self.query)

		proc = subprocess.Popen(['python','test.py',query],shell=False,stdout=subprocess.PIPE)
		
		resp = Response(proc.communicate()[0].decode())
		resp.headers['Access-Control-Allow-Origin'] = '*'
		return resp
		

app = Flask(__name__)
server = EdXServer()

@app.route('/filed',methods=['POST'])
def filer():
	server.get_file()

@app.route('/query',methods=['POST'])
def queried():
	server.get_query


if __name__ == '__main__':
	app.run()
