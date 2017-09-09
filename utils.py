import requests
from bs4 import BeautifulSoup as bs
import numpy as np
def fetch_wikis():
	with open('wiki_links.txt','r') as f:
		for line in f:
			link = line.strip()

def fetch_wiki(url):
	response = requests.get(url)
	soup = bs(response.content,'html.parser')
	print(soup.get_text())
	return
def load_glove():
	glove = {}
	path = "/Users/meshde/Mehmood/EruditeX/data/glove/glove.6B.50d.txt"
	with open(path,'r') as f:
		for line in f:
			l = line.split()
			glove[l[0]] = list(map(float,l[1:]))
	return glove
def get_vector(word,glove):
	try:
		ans = np.array(glove[word]).reshape((1,50))
		return ans
	except:
		return np.random.rand(1,50)
def main():
	url = "https://en.wikipedia.org/wiki/Stanford_University"
	fetch_wiki(url)
	return
if __name__ == '__main__':
	main()
			