import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    
random.shuffle(documents)

# print(documents)
# print('\n------------------------\n')

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print('------------------------\n')

# print(all_words['stupid'])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features ={}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
# print('------------------------\n')

featuressets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuressets[:1900]
testing_set = featuressets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open('Long Version/Pt 15 - Scikit - Learn Incorporation/naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

# save_classifier = open('Long Version/Pt 15 - Scikit - Learn Incorporation/naivebayes.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

print('\nOriginal Naive Bayes Algo accuracy percent :', (nltk.classify.accuracy(classifier, testing_set))*100)
print('------------------------\n')
classifier.show_most_informative_features(15)
print('------------------------\n')

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MNB_classifier accuracy percent :', (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
print('------------------------\n')

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print('GaussianNB_classifier accuracy percent :', (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)
# print('------------------------\n')

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print('BernoulliNB_classifier accuracy percent :', (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
print('------------------------\n')

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print('LogisticRegression_classifier accuracy percent :', (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
print('------------------------\n')

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print('SGDClassifier_classifier accuracy percent :', (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
print('------------------------\n')

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print('SVC_classifier accuracy percent :', (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
print('------------------------\n')

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC_classifier accuracy percent :', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
print('------------------------\n')

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC_classifier accuracy percent :', (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
print('------------------------\n')




'''
    NB :
        - SklearnClassifier() ~> Berfungsi untuk memanggil suatu algoritma classifier.
        - SklearnClassifier(<algoritma classifier>)


'''

