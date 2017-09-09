import subprocess
from bottle import run, post, request, response, get, route

@route('/<path>',method = 'POST')
def process(path):
    output = subprocess.check_output([sys.executable, 'python',path+'.py'],shell=True)
    print(output)
    return output

run(host='localhost', port=8080, debug=True)