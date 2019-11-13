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

def classify(model):
	class1_tokens, class2_tokens = model['class1_fd'], model['class2_fd']

	prior_class1 = return_prior(model, 'class1_count')
	prior_class2 = return_prior(model, 'class2_count')

	print(prior_class1, prior_class2)

def return_prior(model, class_name):
	return math.log(model[class_name]/ \
		(model['class1_count'] + model['class2_count']))

if __name__ == '__main__':
	model = {}
	class1_tokens, class2_tokens = [], []
	test = []
	load_data(model, class1_tokens, class2_tokens, test)
	classify(model)