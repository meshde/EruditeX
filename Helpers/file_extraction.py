# Λnדhгιnכ™
 
import textract


def extract_file_contents(file):

	text = str(textract.process(file))
	# text = textract.process("")

	text = [t for t in text.split('\\n') if t != '']

	print(text)

if __name__ == '__main__':
	extract_file_contents('..\data-og\corpus\sample.docx')