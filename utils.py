import requests
from bs4 import BeautifulSoup as bs
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
	path = ""
	with open(path,'r') as f:
		for line in f:
			l = line.split()
			glove[l[0]] = map(float,l[1:])
	return glove
def get_vector(word,glove):
	return glove[word]
def main():
	url = "https://en.wikipedia.org/wiki/Stanford_University"
	fetch_wiki(url)
	return
if __name__ == '__main__':
	main()
			