
# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend


import pandas as pd
import numpy as np
np.random.seed(1337)
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge,concatenate
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import time

open_bugs_json = 'thesis/fox/deep_data.json'
closed_bugs_json = 'thesis/fox/classifier_data_5.json'

#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

#2. Classifier hyperparameters
numCV = 10
max_sentence_len = 50
min_sentence_length = 15
rankK = 10
batch_size = 32

#========================================================================================
# Preprocess the open bugs, extract the vocabulary and learn the word2vec representation
#========================================================================================
#with open(open_bugs_json) as data_file:
#    data = json.load(data_file, strict=False)

data=pd.read_csv('chro_open.csv',keep_default_na=False)
all_data = []
for i in range(len(data)):
    item=data.iloc[i]
    print(i)
   #1. Remove \r
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
   #2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)
    #3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    #4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title= re.sub(r'(\w+)0x\w+', '', current_title)
    #5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    #6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    #7. Strip trailing punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]
    #8. Join the lists
    current_data = current_title_filter + current_desc_filter
    all_data.append(current_data)

# Learn the word2vec model and extract vocabulary
wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
vocab_size = len(vocabulary)
#========================================================================================
# Preprocess the closed bugs, using the extracted the vocabulary
#========================================================================================
#with open(closed_bugs_json) as data_file:
#    data = json.load(data_file, strict=False)
print('closed')
data=pd.read_csv('chro_5.csv',keep_default_na=False)
all_data = []
all_owner = []
for i in range(len(data)):
    item=data.iloc[i]
    #1. Remove \r
    print(i)
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
    #2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)
    #3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    #4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title= re.sub(r'(\w+)0x\w+', '', current_title)
    #5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    #6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    #7. Strip punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]
    #8. Join the lists
    current_data = current_title_filter + current_desc_filter
    all_data.append(current_data)
    all_owner.append(item['owner'])


#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
totalLength = len(all_data)
splitLength = round(totalLength / (numCV + 1))
seconds=time.time()

for i in range(1, numCV+1):
    # Split cross validation set
    print (i)
    train_data = all_data[:i*splitLength-1]
    test_data = all_data[i*splitLength:(i+1)*splitLength-1]
    train_owner = all_owner[:i*splitLength-1]
    test_owner = all_owner[i*splitLength:(i+1)*splitLength-1]

    # Remove words outside the vocabulary
    updated_train_data = []
    updated_train_data_length = []
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    for j, item in enumerate(train_data):
    	current_train_filter = [word for word in item if word in vocabulary]
    	if len(current_train_filter)>=min_sentence_length:
    	  updated_train_data.append(current_train_filter)
    	  updated_train_owner.append(train_owner[j])

    for j, item in enumerate(test_data):
    	current_test_filter = [word for word in item if word in vocabulary]
    	if len(current_test_filter)>=min_sentence_length:
    	  final_test_data.append(current_test_filter)
    	  final_test_owner.append(test_owner[j])

    # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []
    for j in range(len(final_test_owner)):
    	if final_test_owner[j] not in unwanted_owner:
    		updated_test_data.append(final_test_data[j])
    		updated_test_owner.append(final_test_owner[j])

    unique_train_label = list(set(updated_train_owner))
    classes = np.array(unique_train_label)

    # Create train and test data for deep learning + softmax
    X_train = np.empty(shape=[len(updated_train_data), max_sentence_len, embed_size_word2vec], dtype='float32')
    Y_train = np.empty(shape=[len(updated_train_owner),1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
    for j, curr_row in enumerate(updated_train_data):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X_train[j, sequence_cnt, :] = wordvec_model[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_len-1:
      			        break
        for k in range(sequence_cnt, max_sentence_len):
            X_train[j, k, :] = np.zeros((1,embed_size_word2vec))
        Y_train[j,0] = unique_train_label.index(updated_train_owner[j])

    X_test = np.empty(shape=[len(updated_test_data), max_sentence_len, embed_size_word2vec], dtype='float32')
    Y_test = np.empty(shape=[len(updated_test_owner),1], dtype='int32')
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
    for j, curr_row in enumerate(updated_test_data):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X_test[j, sequence_cnt, :] = wordvec_model[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_len-1:
      			        break
        for k in range(sequence_cnt, max_sentence_len):
            X_test[j, k, :] = np.zeros((1,embed_size_word2vec))
        Y_test[j,0] = unique_train_label.index(updated_test_owner[j])

    y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
    y_test = np_utils.to_categorical(Y_test, len(unique_train_label))
    print(len(unique_train_label))
    import sys
    # Construct the deep learning model
    sequence = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    forwards_1 = LSTM(1024)(sequence)
    after_dp_forward_4 = Dropout(0.20)(forwards_1)
    backwards_1 = LSTM(1024, go_backwards=True)(sequence)
    after_dp_backward_4 = Dropout(0.20)(backwards_1)
    merged=concatenate([after_dp_forward_4,after_dp_backward_4],axis=-1)
#merged = merge([after_dp_forward_4, after_dp_backward_4], mode='concat', concat_axis=-1)
    after_dp = Dropout(0.5)(merged)
    output = Dense(len(unique_train_label), activation='softmax')(after_dp)
    model = Model(input=sequence, output=output)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    hist = model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=200)

    predict = model.predict(X_test)
    accuracy = []
    sortedIndices = []
    pred_classes = []
    correc_pred=[]
    for ll in predict:
   	    sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
    for k in range(1, rankK+1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:
            if k==10:
                correc_pred.append(sortedInd[:k])
            if updated_test_owner[id] in classes[sortedInd[:k]]:
                trueNum += 1
                pred_classes.append(classes[sortedInd[:k]])
            id += 1
        accuracy.append((float(trueNum) / len(predict)) * 100)
    print (accuracy)
    print(time.time()-seconds)
    train_result = hist.history
    name='cv'+str(i)
    pd.DataFrame(correc_pred).to_csv('fast/'+name+'pred.csv')
    pd.DataFrame(classes).to_csv('fast/'+name+'class.csv')
    pd.DataFrame(updated_test_owner).to_csv('fast/'+name+'_correct.csv')




 #   print(train_result)
    del model

#========================================================================================
# Split cross validation sets and perform baseline classifiers
#========================================================================================
'''
#totalLength = len(all_data)
#splitLength = totalLength / (numCV + 1)

for i in range(1, numCV+1):
    # Split cross validation set
    print (i)
    train_data = all_data[:i*splitLength-1]
    test_data = all_data[i*splitLength:(i+1)*splitLength-1]
    train_owner = all_owner[:i*splitLength-1]
    test_owner = all_owner[i*splitLength:(i+1)*splitLength-1]

    # Remove words outside the vocabulary
    updated_train_data = []
    updated_train_data_length = []
    updated_train_owner = []
    final_test_data = []
    final_test_owner = []
    for j, item in enumerate(train_data):
    	current_train_filter = [word for word in item if word in vocabulary]
    	if len(current_train_filter)>=min_sentence_length:
    	  updated_train_data.append(current_train_filter)
    	  updated_train_owner.append(train_owner[j])

    for j, item in enumerate(test_data):
    	current_test_filter = [word for word in item if word in vocabulary]
    	if len(current_test_filter)>=min_sentence_length:
    	  final_test_data.append(current_test_filter)
    	  final_test_owner.append(test_owner[j])

    # Remove data from test set that is not there in train set
    train_owner_unique = set(updated_train_owner)
    test_owner_unique = set(final_test_owner)
    unwanted_owner = list(test_owner_unique - train_owner_unique)
    updated_test_data = []
    updated_test_owner = []
    updated_test_data_length = []
    for j in range(len(final_test_owner)):
    	if final_test_owner[j] not in unwanted_owner:
    		updated_test_data.append(final_test_data[j])
    		updated_test_owner.append(final_test_owner[j])

    train_data = []
    for item in updated_train_data:
    	  train_data.append(' '.join(item))

    test_data = []
    for item in updated_test_data:
    	  test_data.append(' '.join(item))

    vocab_data = []
    for item in vocabulary:
    	  vocab_data.append(item)

    # Extract tf based bag of words representation
    tfidf_transformer = TfidfTransformer(use_idf=False)
    count_vect = CountVectorizer(min_df=1, vocabulary= vocab_data,dtype=np.int32)

    train_counts = count_vect.fit_transform(train_data)
    train_feats = tfidf_transformer.fit_transform(train_counts)
   # print train_feats.shape

    test_counts = count_vect.transform(test_data)
    test_feats = tfidf_transformer.transform(test_counts)
    #print test_feats.shape
    #print "======================="

    # perform classifification
    tit='cv'+str(i)
    for classifier in range(1,2):
        #classifier = 3 # 1 - Niave Bayes, 2 - Softmax, 3 - cosine distance, 4 - SVM

        print (classifier)
        if classifier == 1:
            classifierModel = MultinomialNB(alpha=0.01)
            classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
            predict = classifierModel.predict_proba(test_feats)
            classes = classifierModel.classes_
            pre=[]
            accuracy = []
            sortedIndices = []
            pred_classes = []
            for ll in predict:
                sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
            for k in range(1, rankK+1):
                id = 0
                trueNum = 0
                for sortedInd in sortedIndices:
                    if k==10:
                        pre.append(classes[sortedInd[:k]])
                    if updated_test_owner[id] in classes[sortedInd[:k]]:
                        trueNum += 1
                        pred_classes.append(classes[sortedInd[:k]])
                    id += 1
                accuracy.append((float(trueNum) / len(predict)) * 100)
            pd.DataFrame(updated_test_owner).to_csv('y_test_class_MNB_'+tit+'.csv')
            pd.DataFrame(pre).to_csv('y_pred_class_MNB_'+tit+'.csv')

            print (accuracy)
        elif classifier == 2:
            classifierModel = LogisticRegression(solver='lbfgs', penalty='l2', tol=0.01)
            classifierModel = OneVsRestClassifier(classifierModel).fit(train_feats, updated_train_owner)
            predict = classifierModel.predict(test_feats)
            classes = classifierModel.classes_
            pre=[]
            accuracy = []
            sortedIndices = []
            pred_classes = []
            for ll in predict:
                sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
            for k in range(1, rankK+1):
                id = 0
                trueNum = 0
                for sortedInd in sortedIndices:
                    if k==10:
                        pre.append(classes[sortedInd[:k]])

                    if updated_test_owner[id] in classes[sortedInd[:k]]:
                        trueNum += 1
                        pred_classes.append(classes[sortedInd[:k]])
                    id += 1
                accuracy.append((float(trueNum) / len(predict)) * 100)
            pd.DataFrame(updated_test_owner).to_csv('y_test_class_soft_'+tit+'.csv')
            pd.DataFrame(pre).to_csv('y_pred_class_soft_'+tit+'.csv')

            print (accuracy)
        elif classifier == 3:
            predict = cosine_similarity(test_feats, train_feats)
            classes = np.array(updated_train_owner)
            classifierModel = []

            accuracy = []
            sortedIndices = []
            pred_classes = []
            pre=[]
            for ll in predict:
                sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
            for k in range(1, rankK+1):
                id = 0
                trueNum = 0
                for sortedInd in sortedIndices:
                    if k==10:
                        pre.append(classes[sortedInd[:k]])

                    if updated_test_owner[id] in classes[sortedInd[:k]]:
                        trueNum += 1
                        pred_classes.append(classes[sortedInd[:k]])
                    id += 1
                accuracy.append((float(trueNum) / len(predict)) * 100)
            pd.DataFrame(updated_test_owner).to_csv('y_test_class_cos_'+tit+'.csv')
            pd.DataFrame(pre).to_csv('y_pred_class_cos_'+tit+'.csv')

            print (accuracy)
        elif classifier == 4:
            classifierModel = svm.SVC(probability=True, verbose=False, decision_function_shape='ovr', random_state=42)
            classifierModel.fit(train_feats, updated_train_owner)
            predict = classifierModel.predict(test_feats)
            classes = classifierModel.classes_
            pre=[]
            accuracy = []
            sortedIndices = []
            pred_classes = []
            for ll in predict:
                sortedIndices.append(sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True))
            for k in range(1, rankK+1):
                id = 0
                trueNum = 0
                for sortedInd in sortedIndices:
                    if k==10:
                        pre.append(classes[sortedInd[:k]])

                    if updated_test_owner[id] in classes[sortedInd[:k]]:
                        trueNum += 1
                        pred_classes.append(classes[sortedInd[:k]])
                    id += 1
                accuracy.append((float(trueNum) / len(predict)) * 100)
            pd.DataFrame(updated_test_owner).to_csv('y_test_class_svm_'+tit+'.csv')
            pd.DataFrame(pre).to_csv('y_pred_class_svm_'+tit+'.csv')

            print (accuracy)
