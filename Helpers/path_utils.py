import os

def get_base_path():
	BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	return BASE

def get_sick_path():
	BASE = get_base_path()
	PATH = os.path.join(os.path.join(BASE,'data'),'SICK.txt')
	return PATH
