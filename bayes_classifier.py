import nltk
import math

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

def classify(model, test):
	class1_tokens, class2_tokens = model['class1_fd'], model['class2_fd']

	prior_class1 = return_prior(model, 'class1_count')
	prior_class2 = return_prior(model, 'class2_count')

	answer_key = [n[0] for n in test]
	predictions = []
	miss_count = 0

	out = open('predictions.txt', 'w', encoding="utf-8")

	for index, line in enumerate(test):
		tokens = nltk.word_tokenize(line[1])
		total_class1_p, total_class2_p = prior_class1, prior_class2
		for t in tokens:
			p = math.log((model['class1_fd'][t] + 1)/ \
				(class1_tokens.N() + class1_tokens.B()))
			total_class1_p += p
		for t in tokens:
			p = math.log((model['class2_fd'][t] + 1)/ \
				(class2_tokens.N() + class2_tokens.B()))
			total_class2_p += p
		c = ('ham' if total_class1_p > total_class2_p else 'spam')
		predictions.append(c)
		miss_count += (c != line[0])
		print(c, line[0], index)
		print(c, file=out)

	print((len([n[0] for n in test]) - miss_count)/len([n[0] for n in test]))


def return_prior(model, class_name):
	return math.log(model[class_name]/ \
		(model['class1_count'] + model['class2_count']))

if __name__ == '__main__':
	model = {}
	class1_tokens, class2_tokens = [], []
	test = []
	load_data(model, class1_tokens, class2_tokens, test)
	classify(model, test)