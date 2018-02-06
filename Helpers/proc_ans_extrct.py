import xml.etree.ElementTree
import json
import pickle
import os

def _process_ans_extract_dataset():

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/ans_extraction.xml')
	e = xml.etree.ElementTree.parse(file).getroot()

	qalist = []

	index = 0
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


def json_write(qalist):

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/ans_extrct_data.json')
	with open(file, 'w') as fp:
		for qa in qalist:
			json.dump(qa, fp)

def ds_pickle(qalist):

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/ans_extrct_data.pkl')
	with open(file, 'wb') as fp:
		pickle.dump(qalist, fp)


def get_ans_ext_list():

	file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/ans_extraction/ans_extrct_data.pkl')

	if os.path.isfile(file):
		with open(file, 'rb') as fp:
			qalist = pickle.load(fp)
	
	else:
		qalist = _process_ans_extract_dataset()
		ds_pickle(qalist)

	return qalist


if __name__ == '__main__':
	
	qalist = _process_ans_extract_dataset()
	json_write(qalist)
	get_ans_ext_list()
	# ds_pickle(qalist)
	print(qalist)