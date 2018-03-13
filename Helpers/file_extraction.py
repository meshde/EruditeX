# import textract

def extract_file_contents(file):

	# text = str(textract.process(file))
	
	with open(file) as source_file:
		text = source_file.readlines()

	text = [t for t in text.split('\\n') if t != '']

	# print(text)
	return text

if __name__ == '__main__':
	extract_file_contents('..\data-og\corpus\cricket.docx')