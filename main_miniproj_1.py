import os
from pprint import pprint

from matplotlib import pyplot as plt
import sklearn
from sklearn import datasets

docs_to_train =sklearn.datasets.load_files("/Users/salarjaberi/PycharmProjects/FRSF/BBC", description=None, categories=None, load_content=True, shuffle=False, encoding='latin1', decode_error='strict', random_state=0)

fileCounter = 0
for root, dirs, files in os.walk("/Users/salarjaberi/PycharmProjects/FRSF/BBC/business"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter += 1
    print(fileCounter)

fileCounter1 = 0
for root, dirs, files in os.walk("/Users/salarjaberi/PycharmProjects/FRSF/BBC/entertainment"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter1 += 1
    print(fileCounter1)
fileCounter2 = 0
for root, dirs, files in os.walk("/Users/salarjaberi/PycharmProjects/FRSF/BBC/politics"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter2 += 1
    print(fileCounter2)

fileCounter3 = 0
for root, dirs, files in os.walk("/Users/salarjaberi/PycharmProjects/FRSF/BBC/sport"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter3 += 1
    print(fileCounter3)

fileCounter4 = 0
for root, dirs, files in os.walk("/Users/salarjaberi/PycharmProjects/FRSF/BBC/tech"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter4 += 1
    print(fileCounter4)

x=["Folder1 ","Folder 2","Folder 3","Folder 4"]
y=[fileCounter,fileCounter1,fileCounter3,fileCounter4]
plt.bar(x,y)
f=plt.figure()
plt.show()


corpus=(list(docs_to_train.data))
#pprint(corpus)
from sklearn.feature_extraction.text import CountVectorizer
#list of text documents
text = corpus
# # create the transform
vectorizer = CountVectorizer()
# # tokenize and build vocab
vectorizer.fit(text)
# # summarize
print(vectorizer.vocabulary_)
# # encode document
vector = vectorizer.transform(text)
# # summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())