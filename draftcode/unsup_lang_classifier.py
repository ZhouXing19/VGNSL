'''
https://towardsdatascience.com/applying-machine-learning-to-classify-an-unsupervised-text-document-e7bb6265f52#:~:text=K%2Dmeans%20is%20one%20of,centroids%2C%20one%20for%20each%20cluster.
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


en_path = "/Users/zhou/Dropbox/Travail/MAThesis/VGNSL/data/mscoco/en/train_caps.txt"
fr_path = "/Users/zhou/Dropbox/Travail/MAThesis/VGNSL/data/mscoco/fr/train_caps.txt"

en_test_path = "/Users/zhou/Dropbox/Travail/MAThesis/VGNSL/data/mscoco/en/test_caps.txt"
fr_test_path = "/Users/zhou/Dropbox/Travail/MAThesis/VGNSL/data/mscoco/fr/test_caps.txt"

en_doc = []
with open(en_path, 'r') as en_file:
    for en_line in en_file.readlines():
        en_doc.append(en_line.strip('\n'))

en_file.close()

fr_doc = []
with open(fr_path, 'r') as fr_file:
    for fr_line in fr_file.readlines():
        fr_doc.append(fr_line.strip('\n'))


fr_file.close()

en_test_doc = []
with open(en_test_path, 'r') as en_file:
    for en_line in en_file.readlines():
        en_test_doc.append(en_line.strip('\n'))
en_file.close()

fr_test_doc = []
with open(fr_test_path, 'r') as fr_file:
    for fr_line in fr_file.readlines():
        fr_test_doc.append(fr_line.strip('\n'))

fr_file.close()


document = en_doc + fr_doc

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(document)
true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


for i in range(true_k):
 print("Cluster %d:" % i)
 for ind in order_centroids[i, :10]:
    print("%s" % terms[ind])


'''
fr : 1
en: 0
'''

fr_error = 0
en_error = 0

for fr_test_sent in fr_test_doc:
    fr_X = vectorizer.transform([fr_test_sent])
    fr_predicted = model.predict(fr_X)
    if fr_predicted != 1:
        print(fr_test_sent)
        fr_error += 1


for en_test_sent in en_test_doc:
    en_X = vectorizer.transform([en_test_sent])
    en_predicted = model.predict(en_X)
    if en_predicted != 0:
        en_error += 1





