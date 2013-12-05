from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import numpy as np
from pandas import *
import matplotlib.pyplot as plt

from scratch import *

# label description for data
label_desc = [
	"s1,I can't tell",	
	"s2,Negative",
	"s3,Neutral / author is just sharing information",
	"s4,Positive",
	"s5,Tweet not related to weather condition",

	"w1,current (same day) weather",
	"w2,future (forecast)",
	"w3,I can't tell",
	"w4,past weather",

	"k1,clouds",
	"k2,cold",
	"k3,dry",
	"k4,hot",
	"k5,humid",
	"k6,hurricane",
	"k7,I can't tell",
	"k8,ice",
	"k9,other",
	"k10,rain",
	"k11,snow",
	"k12,storms",
	"k13,sun",
	"k14,tornado",
	"k15,wind",
]
label_name = [
	"s1","s2","s3","s4","s5","w1","w2","w3","w4","k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11","k12","k13","k14","k15"
]

def tokenizer(text):
	tok = nltk.tokenize.RegexpTokenizer(r'\w{3,}')
	stopwords = nltk.corpus.stopwords.words('english')
	return [w.lower() for w in tok.tokenize(text) if w.lower() not in stopwords]

###############################################################
# Routine Utilities
###################
def read_labeled_dataset(input_filename):
    # Read files
    tweets = read_tweets(input_filename)
    labels = read_labels(input_filename)        

    # Get feature vectors by tf-idf
    vec = TfidfVectorizer(tokenizer=tokenizer, max_features=100)
    data = vec.fit_transform(tweets).toarray()

    return np.array(data), np.array(labels)

def read_labels(input_filename):
	labels = read_labels(input_filename)
	return labels

def get_features_from_dataset(input_filename):
    # Read files
    tweets = read_tweets(input_filename)

    # Get feature vectors by tf-idf
    vec = TfidfVectorizer(tokenizer=tokenizer, max_features=100)
    data = vec.fit_transform(tweets).toarray()

    ### Additional features
    
    add_features = []
    important_words = ['sunny', 'windy', 'humid', 'hot', 'heat', 'cold', 'dry', 'ice', 'icy', 'rain', 'rainy', 'snow', 'tornado']

    """
    for twtr in tweets:
    	# Important words
    	important_words_ftr = [int(word in twtr) for word in important_words]
    	# Tense
    	tagz = zip(*nltk.pos_tag(nltk.word_tokenize(twtr)))[1]
    	past_num = len([v for v in tagz if v == 'VBD'])
    	present_num = len([v for v in tagz if v in ['VBP', 'VB']])
    	
    	add_features.append(important_words_ftr + [past_num, present_num])

    data_extended = np.hstack((data, add_features))	
    column_names = vec.get_feature_names() + ['add_words:' + word for word in important_words] + ['past_tense_num', 'present_tense_num']
    features = DataFrame(data_extended, columns = column_names)
    """
    for twtr in tweets:
    	# Important words
    	important_words_ftr = [int(word in twtr) for word in important_words]
    	
    	add_features.append(important_words_ftr)

    add_features = np.array(add_features)

    data_extended = np.hstack((data, add_features))
    column_names = vec.get_feature_names() + ['add_words:' + word for word in important_words]
    features = DataFrame(data_extended, columns = column_names)
    
    return features

def read_all_dataset(train_filename, test_filename = ""):
	# Read files
	train_tweets = read_tweets(train_filename)
	labels = read_labels(train_filename)	

	if test_filename != "":
		test_tweets = []
		test_ids = []
		with open(test_filename, 'r') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			next(spamreader, None) # skip the header
			for row in spamreader:
				test_tweets.append(row[1])
				test_ids.append(row[0])

		all_tweets = train_tweets + test_tweets
		n_train = len(train_tweets)
		n_test = len(test_tweets)

		# Get feature vectors by tf-idf
		vec = TfidfVectorizer(tokenizer=tokenizer, max_features=200)
		all_data = vec.fit_transform(all_tweets).toarray()
		train_data = all_data[0:n_train]
		test_data = all_data[n_train:]

		return np.array(train_data), np.array(labels), np.array(test_data), np.array(test_ids)

# Train model
def train_single_model(train_data, train_labels, algo):
	"""
	Train the model for a single label dimension
	"""
	if algo == 'svr_rbf':
		"""
		SVM regression, RBF kernel
		"""
		svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		svr_rbf.fit(train_data, train_labels)
		return svr_rbf

	if algo == 'svr_lin':
		"""
		SVM regression, linear
		"""
		svr_lin = SVR(kernel='linear')
		svr_lin.fit(train_data, train_labels)
		return svr_lin

	if algo == 'ridge':
		"""
		Ridge regression
		"""
		clf = Ridge(alpha = 0.5)
		clf.fit(train_data, train_labels)
		return clf

	# No hit algorithm
	print "unimplemented model type"
	return None

# Train fused models
def train_model(train_data, train_labels):
	"""
	Train models as a whole
	"""
	# VERSION 1
	# train a single model for each label dimension
	models = []
	label_ver = zip(*train_labels)
	for label in label_ver:
		model = train_single_model(train_data, label, 'ridge')
		models.append(model)

	return models

# Get prediction
def get_prediction(fused_model, test_data):
	fused_prediction = []
	for model in fused_model:
		fused_prediction.append(model.predict(test_data))

	return np.array(fused_prediction).T

def evaluate_result(pred, test_labels):
	"""
	Evaluate the result. 
	Print stats description and return the error series
	"""
	npred = np.float32(np.array(pred))
	nlabels = np.float32(np.array(test_labels))

	error = DataFrame(np.abs(nlabels - npred), columns=label_desc)
	#for i in range
	print "The global RMSE is ", error.apply(lambda x: np.sqrt(x.dot(x)), axis=0) / np.sqrt(len(error))

	return error

################################################################
# Main routines
###############
def routine_test(train_filename, algo):
	"""
	Used to test the algorithm.
	Evaluation is printed out in this method
	"""
	# Calculate time
	import time
	start_time = time.time()

	# Prepare data
	data, labels = read_labeled_dataset(train_filename)
	# Seperate the test and train
	fold = int(len(data) * 0.9)
	train_data = data[0:fold]
	train_labels = labels[0:fold]
	test_data = data[fold+1:]
	test_labels = labels[fold+1:]
	
	# Train model
	fused_model = train_model(train_data, train_labels)

	# Get predict result
	result = get_prediction(fused_model, test_data)

	# Evaluate result
	error_df = evaluate_result(result, test_labels)

	# Calculate time
	print 'Execution time: ', time.time() - start_time, 'seconds.'

	return error_df

def routine_work(train_filename, test_filename, algo):
# Calculate time
	import time
	start_time = time.time()

	# Prepare data
	data, labels, test_data, test_ids = read_all_dataset(train_filename, test_filename)
	
	# Train model
	fused_model = train_model(data, labels)

	# Get predict result
	result = get_prediction(fused_model, test_data)

	# Write result
	result_df = DataFrame(result, columns=label_name)
	result_df.insert(0, 'id', test_ids)
	result_df.to_csv('../submission/submit.csv', index=False, float_format='%.3f')

	# Calculate time
	print 'Execution time: ', time.time() - start_time, 'seconds.'


if __name__ == 'main':
	routine_test('../data/train_1000.csv')
