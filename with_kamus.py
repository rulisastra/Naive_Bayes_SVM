from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import string
import re
import time
import numpy as np
from datetime import datetime as dt
import statistics as stat
import collections
from sklearn.pipeline import Pipeline
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold

set_input = 'train.xlsx'

# read data
data = pd.read_excel(set_input)
kamus = pd.read_csv('kamus.csv')['A'].array
slang = pd.read_csv('slang.csv')

# create stemmer
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()

# create stopword
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

punc = string.punctuation.replace(':','')


start_prep = dt.now()

# replace slang
def replace_slang(string_to_be_replaced):
    clean_ = ''
    for j in string_to_be_replaced.split(' '):
        if j in slang['tidak_baku'].values:
            x = int(np.where(slang['tidak_baku'].values == j)[0])
            clean_ = clean_ + ' ' + (slang['baku'].values[x])

        else:
            clean_ = clean_ + ' ' + j
    return clean_
#data_after_remove = ' '.join(clean_)

data_after_remove = []
for i in data['reply']:
    i = i.replace(':', '')
    print('before prep: ', i)
    # remove punctuation except :
    i = i.translate(str.maketrans('', '', punc))
    i = i.lower()   

    # stem
    i  = re.sub(r"http\S+", "", i)
    i = replace_slang(i)
    i = stemmer.stem(i)

    clean_one = ' '.join(i)
    print('proses prep: ', i)
    
    # remove nonalpha num, except space and :
    clean_one = re.sub(r'[^a-zA-Z0-9 :]','', clean_one)

    # cek di kamus
    clean_one = ' '.join([x for x in clean_one.split(' ') if x in kamus])

    # remove stopwords
    clean_one = stopword.remove(i)

    print('after prep: ', clean_one)

    data_after_remove.append(clean_one)

elapsed_prep = dt.now() - start_prep
elapsed_prep = elapsed_prep.total_seconds()

kf = KFold(n_splits=10, shuffle=True)

# time start run mnb
start_mnb = dt.now()

clf1 = MultinomialNB()

# vectorizer mnb
vectorizer1 = TfidfVectorizer()
X1 = vectorizer1.fit_transform(data_after_remove)
X1 = X1.toarray()

vectorizer1_label = CountVectorizer()
y1 = vectorizer1_label.fit_transform(data['value'])

y1 = data['value']
kf.get_n_splits(X1)

# start mnb
acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for train_index, test_index in kf.split(X1):
    X_train, X_test = X1[train_index], X1[test_index]
    y_train, y_test = y1[train_index], y1[test_index]

    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    print('jumlah prediksi MNB: ', collections.Counter(y_pred))

    y_test_num = []
    y_pred_num = []
    for i in range(len(y_pred)):
        if y_pred[i] == 'positif':
            y_pred_num.append(1)
        elif y_pred[i] == 'negatif':
            y_pred_num.append(2)
        elif y_pred[i] == 'netral':
            y_pred_num.append(0)

        if y_test.array[i] == 'positif':
            y_test_num.append(1)
        elif y_test.array[i] == 'negatif':
            y_test_num.append(2)
        elif y_test.array[i] == 'netral':
            y_test_num.append(0)

    acc_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test_num, y_pred_num, average='macro'))
    recall_scores.append(recall_score(y_test_num, y_pred_num, average='macro'))
    f1_scores.append(f1_score(y_test_num, y_pred_num, average='macro'))


print('\nmean accuracy_score MNB', stat.mean(acc_scores)*100,'%')
print('mean precision_scores MNB', stat.mean(precision_scores)*100,'%')
print('mean recall_scores MNB', stat.mean(recall_scores)*100,'%')
print('mean f1_scores MNB', stat.mean(f1_scores)*100,'%')

elapsed_mnb = dt.now() - start_mnb
print('Waktu run MNB: %s s' % str(elapsed_mnb.total_seconds() + elapsed_prep))

# time start run svm
start_svm = dt.now()

clf2 = LinearSVC(random_state=0, tol=1e-5)

# vectorizer svm
vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(data_after_remove)

X2 = X2.toarray()
y2 = data['value_num'].array
kf.get_n_splits(X2)

# start svm
acc_scores2 = []
precision_scores2 = []
recall_scores2 = []
f1_scores2 = []
for train_index, test_index in kf.split(X2):
    X_train, X_test = X2[train_index], X2[test_index]
    y_train, y_test = y2[train_index], y2[test_index]

    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    print('jumlah prediksi SVM: ', collections.Counter(y_pred))

    acc_scores2.append(accuracy_score(y_test, y_pred))
    precision_scores2.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores2.append(recall_score(y_test, y_pred, average='macro'))
    f1_scores2.append(f1_score(y_test, y_pred, average='macro'))

print('\nmean accuracy_score SVM', stat.mean(acc_scores2)*100,'%')
print('mean precision_scores SVM', stat.mean(precision_scores2)*100,'%')
print('mean recall_scores SVM', stat.mean(recall_scores2)*100,'%')
print('mean f1_scores SVM', stat.mean(f1_scores2)*100,'%')

elapsed_svm = dt.now() - start_svm
print('Waktu run SVM: %s s' % str(elapsed_svm.total_seconds() + elapsed_prep))