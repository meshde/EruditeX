import xml.etree.ElementTree
import json
import pickle
import os

def _process_ans_extract_dataset():

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/ans_extraction.xml')
	e = xml.etree.ElementTree.parse(file).getroot()

	qalist = []

	for qapairs in e.findall('qapairs'):

		qa = {}
		for child in qapairs:
		
			if child.tag == 'question':
				# print(child.text)
				qa['qstn'] = child.text.split('?')[0].replace('\n', '').replace(' ', '').replace('\t', ' ')

			elif child.tag == 'positive':
				child
				qa['ans_sent'] = child.text.split('.')[0].replace('\n', '').replace(' ', '').replace('\t', ' ')
				qa['ans'] = child.text.split('\n')[-3].replace('\t', ' ')
				# qa[ans_phrase] = ''

		qalist.append(qa)
	return qalist


def _process_babi(source):

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/en/'+source+'.txt')
	qalist = []
	ans_sents = []
	qa = {}

	with open(file) as fp:
		data = fp.readlines()
		for line in data:
			if line.split(' ', 1)[0] == '1':
				# print(ans_sents)
				ans_sents = []

			ans_sents.append(line.split(' ', 1)[1][:-2])
			if '?' in line:
				line = line.split('\t')
				# print(line)
				qa['qstn'] = line[0].split(' ', 1)[1]
				qa['ans_sent'] = ans_sents[int(line[2].replace('\n',''))-1]
				qa['ans'] = line[1]
				qalist.append(qa)
				# print(qa)
				qa = {}

	return qalist


def json_write(qalist, source):

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/'+source+'_extract_data.json')
	with open(file, 'w') as fp:
		for qa in qalist:
			json.dump(qa, fp)

def ds_pickle(qalist, source):

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/'+source+'_extract_data.pkl')
	
	with open(file, 'wb') as fp:
		pickle.dump(qalist, fp)


def get_ans_ext_list(source):

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/'+source+'_extract_data.pkl')

	if os.path.isfile(file):
		with open(file, 'rb') as fp:
			qalist = pickle.load(fp)
	
	else:
		qalist = _process_ans_extract_dataset()
		ds_pickle(qalist)

	return qalist


if __name__ == '__main__':
	
	# qalist = _process_ans_extract_dataset()
	qalist = _process_babi('qa1_single-supporting-fact_train')
	# qalist = _process_babi('qa2_two-supporting-facts_test')

	# json_write(qalist)
	# get_ans_ext_list()
	ds_pickle(qalist, 'babi_qa1')

	# print(qalist)