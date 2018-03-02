import os

def get_base_path():
	BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	return BASE

def get_sick_path():
	BASE = get_base_path()
	PATH = os.path.join(BASE,'data/cache/SICK_cache.pkl')
	return PATH

def get_babi_ans_extract_path(extension="json"):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'data/ans_extraction/babi/{}/'.format(extension))
    return PATH

def get_babi_ans_extract_input_path():
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'data/cache/babi_ans_extraction.pkl')
    return PATH

def get_logs_path(file_path):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'logs/'+file_path)
    return PATH

def get_save_states_path(file_path):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'states/'+file_path)
    return PATH



