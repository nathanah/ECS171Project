import nltk

def load_data(model, class1_tokens, class2_tokens, test):
	file = open('SMSSpamCollection', 'r', encoding='utf_8')
	model['class1_count'], model['class2_count'] = 0,0
	for index, line in enumerate(file):
		l = line.split("\t")
		if index >= 4000:
			test.append(l)
		else:
			if l[0] == "ham":
				class1_tokens += nltk.word_tokenize(l[1])
				model['class1_count'] += 1
			elif l[0] == "spam":
				class2_tokens += nltk.word_tokenize(l[1])
				model['class2_count'] += 1

	model['class1_fd'] = nltk.FreqDist(class1_tokens)
	model['class2_fd'] = nltk.FreqDist(class2_tokens)

if __name__ == '__main__':
	model = {}
	class1_tokens, class2_tokens = [], []
	test = []
	load_data(model, class1_tokens, class2_tokens, test)