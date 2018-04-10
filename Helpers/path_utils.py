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

def get_babi_ans_mod_path():
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'data/cache/babi_ans_model.pkl')
    return PATH

def get_logs_path(file_path):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'logs/'+file_path)
    return PATH

def get_save_states_path(file_path):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'states/'+file_path)
    return PATH

def get_config_path():
    BASE = get_base_path()
    CONFIG_PATH = os.path.join(BASE, 'Configurations')
    return CONFIG_PATH

def get_config_file_path(filename):
    CONFIG_PATH = get_config_path()
    config_file_path = os.path.join(CONFIG_PATH, filename)
    return config_file_path

def get_cache_path():
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'data/cache/')
    return PATH

def get_wikiqa_raw_path(mode):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'data/wikiqa/WikiQA-{}.tsv'.format(mode))
    return PATH

def get_babi_raw_path(babi_name, mode):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'data/en/{}{}.txt'.format(babi_name, mode))
    return PATH

def get_summary_path(model, mode):
    BASE = get_base_path()
    PATH = os.path.join(BASE, 'data/visualize/{}/{}/'.format(model, mode))
    return PATH

def get_squad_path(mode):
    BASE = get_base_path()
    if mode == 'train':
        filename = 'train-v1.1.json'
    else:
        filename = 'dev-v1.1.json'
    PATH = os.path.join(BASE, 'data/squad/{}'.format(filename))
    return PATH
