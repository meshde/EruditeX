from os import path


def process_babi_for_abcnn(babi):
	from tqdm import tqdm

	samples = []
	
	for line in tqdm(babi, total=len(babi), ncols=75, unit='Line'):
		line_number, data = tuple(line.split(' ', 1))
		data = data.lower()
		if line_number == '1':
			context = []
			line_numbers = []
		
		if '?' in data:
			question, ans_token, support = tuple(data.split(sep='\t'))
			l_n = list(line_numbers)
			c = list(context)
			samples.append((l_n, c, question, ans_token, support))
		else:
			context.append(data)
			line_numbers.append(line_number)

	return samples

def get_babi_raw_for_abcnn(babi_id, mode='test'):
	babi_raw = []

	babi_map = {
	'1': 'qa1_single-supporting-fact_',
	'2': 'qa2_two-supporting-facts_',
	'3': 'qa3_three-supporting-facts_',
	'4': 'qa4_two-arg-relations_',
	'5': 'qa5_three-arg-relations_',
	'6': 'qa6_yes-no-questions_',
	'7': 'qa7_counting_',
	'8': 'qa8_lists-sets_',
	'9': 'qa9_simple-negation_',
	'10': 'qa10_indefinite-knowledge_',
	'11': 'qa11_basic-coreference_',
	'12': 'qa12_conjunction_',
	'13': 'qa13_compound-coreference_',
	'14': 'qa14_time-reasoning_',
	'15': 'qa15_basic-deduction_',
	'16': 'qa16_basic-induction_',
	'17': 'qa17_positional-reasoning_',
	'18': 'qa18_size-reasoning_',
	'19': 'qa19_path-finding_',
	'20': 'qa20_agents-motivations_'
	}
	babi_name = babi_map[babi_id]
	babi_file = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'data/en/{}test.txt'.format(babi_name))
	
	with open(babi_file, 'r') as f:
		print('  > Getting raw bAbI {} {}'.format(babi_id, mode))
		for line in f:
			babi_raw.append(line.replace('\n', ''))
	
	return babi_raw

def get_babi(babi_id='1'):
	babi_raw = get_babi_raw_for_abcnn(babi_id)
	babi_data = process_babi_for_abcnn(babi_raw)

	final_dict = []
	for sample in babi_data:
		l_n, c, question, ans_token, support = sample
		context = ' '.join(c)
		d = {}
		d['context'] = context
		d['question'] = question
		d['ans_token'] = ans_token
		d['support'] = support
		d['line_numbers'] = l_n
		final_dict.append(d)

	return final_dict
