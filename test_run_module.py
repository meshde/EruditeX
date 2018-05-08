import subprocess
import shlex

def test(parameters):

	print("Testing with %d parameters"%(len(parameters)))
	print("Parameters being passed: ", parameters)
	params = ['./run.sh']
	params.extend(parameters)
	# print(params)
	val = subprocess.call(params)
	print("\n***Test case executed")

	

if __name__ == '__main__':
	# test(["install_packages"])
	# test(["grumpy"])
	# test([""])
	# test(["train_dtrnn","50","100","10"])
	# test(["train_ans_extract","100"])
	test(["train_ans_extract","100","200"])



