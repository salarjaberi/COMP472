import os
import pathlib
from turtle import pd
from matplotlib import pyplot as plt
from pathlib import Path
import sklearn
import pandas as pd
from sklearn import datasets
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

docs_to_train =sklearn.datasets.load_files(r"C:\Users\faraj\PycharmProjects\mini-project 1\BBC", description=None, categories=None, load_content=True, shuffle=False, encoding='latin1', decode_error='strict', random_state=0)

fileCounter = 0
for root, dirs, files in os.walk(r"C:\Users\faraj\PycharmProjects\mini-project 1\BBC\business"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter += 1
    #print(fileCounter)
fileCounter1 = 0
for root, dirs, files in os.walk(r"C:\Users\faraj\PycharmProjects\mini-project 1\BBC\entertainment"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter1 += 1
    #print(fileCounter1)
fileCounter2 = 0
for root, dirs, files in os.walk(r"C:\Users\faraj\PycharmProjects\mini-project 1\BBC\politics"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter2 += 1
    #print(fileCounter2)
fileCounter3 = 0
for root, dirs, files in os.walk(r"C:\Users\faraj\PycharmProjects\mini-project 1\BBC\sport"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter3 += 1
    #print(fileCounter3)
fileCounter4 = 0
for root, dirs, files in os.walk(r"C:\Users\faraj\PycharmProjects\mini-project 1\BBC\tech"):
    for file in files:
        if file.endswith('.txt'):
            fileCounter4 += 1
    #print(fileCounter4)

x=["business","entertainment","politics","sport","tech"]
y=[fileCounter,fileCounter1,fileCounter2,fileCounter3,fileCounter4]
plt.bar(x,y)
plt.show()




print(docs_to_train)
pprint(list(docs_to_train.filenames))
corpus=(list(docs_to_train.data))



document = corpus

# Create a Vectorizer Object
vectorizer = CountVectorizer()

vectorizer.fit(document)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", vectorizer.vocabulary_)

# Encode the Document
vector = vectorizer.transform(document)

# Summarizing the Encoded Texts
#print("Encoded Document is:")
#print(vector.toarray())



df = pd.DataFrame(vector.toarray())
df.to_csv('data.csv')


#split the data into train and test set
train,test = train_test_split(df, test_size=0.20, random_state=None)
#save the data
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)

nb = MultinomialNB()

# Fit the model
nb.fit(train, test)

# Print the accuracy score
print("Accuracy:",nb.score(train, test))