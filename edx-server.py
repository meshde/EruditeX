import subprocess
from flask import Flask
from flask import Response
from flask import request
import json

app = Flask(__name__)

@app.route('/',methods=['POST'])
def get_passage():
	query = request.get_json(force=True)['query']
	print(query)
#  	resp = Response(query)
	proc = subprocess.Popen(['python','IR/InfoRet.py',query],shell=False,stdout=subprocess.PIPE)
	resp = Response(proc.communicate()[0].decode())
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp
@app.route('/babi',methods=['POST'])
def get_answer():
	query = request.get_json(force=True)['query']
	print(query)
#  	resp = Response(query)
	proc = subprocess.Popen(['python','test.py',query],shell=False,stdout=subprocess.PIPE)
	resp = Response(proc.communicate()[0].decode())
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp

if __name__ == '__main__':
	app.run()
