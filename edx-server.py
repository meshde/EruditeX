import subprocess
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/',methods=['POST'])
def get_passage():
	query = request.form['query']
	proc = subprocess.Popen(['python','IR/InfoRet.py',query],shell=False,stdout=subprocess.PIPE)
	return proc.communicate()[0].decode()

if __name__ == '__main__':
	app.run()