import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import collections
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from scipy import stats
from scipy.sparse import csr_matrix

########## Loading data ###########
def load_train():
	train_file = "train.json"
	train = pd.read_json(train_file)
	X = train.drop('interest_level', axis = 1)
	Y = train['interest_level']
	Y.columns = ['interest_level']
	print "Training data loaded"
	return X,Y, train

def load_test():
	test_file = "test.json"
	X_test = pd.read_json(test_file)
	print "Test dataset loaded"
	return X_test

########## Feature engineering ###########
def clean_date(X):
	X['created'] = pd.to_datetime(X['created'])
	X['year'] = X['created'].dt.year
	X['month'] = X['created'].dt.month
	X['day'] = X['created'].dt.day
	X['hour'] = X['created'].dt.hour
	return X

def num_photos(X):
	num_photos = []
	for photo in X['photos']:
		num_photos.append(len(photo))
	num_photos_df = pd.DataFrame(num_photos, index = X.index, columns = ['num_photos'])
	X = X.assign(num_photos = num_photos_df)
	return X
	
def len_description(X):
	len_description = []
	for description in X['description']:
		len_description.append(len(description))
	len_description_df = pd.DataFrame(len_description, index = X.index, columns = ['len_description'])
	X = X.assign(len_description = len_description_df)
	return X
	
def num_features(X):
	num_features = []
	for feature in X['features']:
		num_features.append(len(feature))
	num_features_df = pd.DataFrame(num_features, index = X.index, columns = ['num_features'])
	X = X.assign(num_features = num_features_df)
	return X
	
def feature_analysis(X):
	vectorizer = CountVectorizer(stop_words = 'english', max_features = 200)	
	features_sparse_df = X["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
	features_sparse = vectorizer.fit_transform(features_sparse_df).toarray()

	return features_sparse
	
def drop_parameters(X):
	X = X.drop('building_id', axis = 1)
	X = X.drop('created', axis = 1)
	X = X.drop('description', axis = 1)
	X = X.drop('display_address', axis = 1)
	X = X.drop('features', axis = 1)
	X = X.drop('manager_id', axis = 1)
	X = X.drop('photos', axis = 1)
	X = X.drop('street_address', axis = 1)
	X = X.drop('listing_id', axis = 1)
	return X
	
def label_cleaner(train, label):
	train[train == 'low'] = label[0]
	train[train == 'medium'] = label[1]
	train[train == 'high'] = label[2]
	return train

def df_np(X_train, X_val, Y_train, Y_val):
	X_train_n = np.asarray(X_train, dtype = 'int')
	Y_train_n = np.asarray(Y_train, dtype = 'int')
	X_val_n = np.asarray(X_val, dtype = 'int')
	Y_val_n = np.asarray(Y_val, dtype = 'int')
	return X_train_n, Y_train_n, X_val_n, Y_val_n	

def modelling(X_train_n, Y_train_n, X_val_n, Y_val_n, X_test_n):
	
	## OneVsRest - Logistic Regression ##
	estimator = OneVsRestClassifier(LogisticRegression())
	print "Training Logistic regression"
	estimator.fit(X_train_n, Y_train_n)
	pred_logit_t = estimator.predict_proba(X_train_n)
	pred_logit_v = estimator.predict_proba(X_val_n)
	print "Log loss on Train (logit):", round(log_loss(Y_train_n, pred_logit_t), 2)
	print "Log loss on Val (logit):", round(log_loss(Y_val_n, pred_logit_v), 2)	
	
	## Random Forests ##
	clf_RF = RandomForestClassifier(n_estimators=100)
	print "Training Random Forest classifier"
	clf_RF.fit(X_train_n, Y_train_n)
	print "Calculating cost on train and validation"
	pred_RF_v = clf_RF.predict_proba(X_val_n)
	pred_RF_t = clf_RF.predict_proba(X_train_n)
	print "Log loss on val (Random Forest):", log_loss(Y_val_n, pred_RF_v)
	print "Log loss on train (Random Forest):", log_loss(Y_train_n, pred_RF_t)
	print "Accuract on val (Random Forest):", clf_RF.score(X_val_n, Y_val_n)
	print "Making predictions.."
	prediction = clf_RF.predict_proba(X_test_n)
	print "Predictions made"
	return pred_logit_t, prediction
	
def submit(prediction, test):
	sub = pd.DataFrame()
	sub = pd.DataFrame(prediction, columns = ['low', 'medium', 'high'])
	print "Top 10 predictions: "
	print sub.head(20)
	print "Listing IDs"
	print sub.head(20)
	sub['listing_id'] = test.listing_id.values
	sub.to_csv('prediction.csv', sep=',', index = False)	
	print "Predictions saved"

	
if __name__ == '__main__':
	X,Y, train = load_train()
	X_test = load_test()
	test = X_test
	X = clean_date(X)
	X_test = clean_date(X_test)
	X = num_photos(X)
	X_test = num_photos(X_test)
	X = len_description(X)
	X_test = len_description(X_test)
	X = num_features(X)
	X_test = num_features(X_test)
	features_sparse = feature_analysis(X)
	features_sparse_test = feature_analysis(X_test)
	X = drop_parameters(X)
	X_test = drop_parameters(X_test)

	X = np.asarray(X, dtype = 'int')
	X = np.hstack((X, features_sparse))
	X_test_n = np.asarray(X_test, dtype = 'int')
	X_test_n = np.hstack((X_test, features_sparse_test))	

	Y_label = [0,1,2]
	Y = label_cleaner(Y, Y_label)
	Y = np.asarray(Y, dtype = 'int')
	#Splitting test and train data
	test_size = 0.3
	seed = 7
	X_train_n, X_val_n, Y_train_n, Y_val_n = train_test_split(X, Y, test_size = test_size, random_state = seed)	

	
	#Modeling and predictions
	#X_test_n = np.asarray(X_test, dtype = 'int')
	#X_train_n, Y_train_n, X_val_n, Y_val_n = df_np(X_train, X_val, Y_train, Y_val)
	pred_high_t, prediction = modelling(X_train_n, Y_train_n, X_val_n, Y_val_n, X_test_n)
	
	#Making a submission of results
	submit(prediction, test)
	
	
	
	
